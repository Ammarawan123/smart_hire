"""test_v59.py — validation of the V59 behaviour layer.  Run next to the V58 files:  python -m unittest test_v59 -v
Synthetic checks validate MACHINERY (maths, guarantees, plumbing). They are not evidence that any cue detects deception."""
import sys, time, threading, json, unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from shared.paths import bootstrap  # noqa: E402
bootstrap()

import behavior_config_v59 as C
import fusion_v59 as F
import text_cues_v59 as T
import voice_cues_v59 as V
import vision_extras_v59 as X
import claim_extractor_v59 as CE
import tts_timed_v59 as TT
import eval_harness_v59 as E
import sim_study_v59 as S


class TestFusionMath(unittest.TestCase):
    def test_baseline_shrinkage_small_n_gives_small_z(self):
        b1, b8 = F.PersonBaseline(0, 1), F.PersonBaseline(0, 1)
        for x in (0.1, -0.1): b1.add(x)
        for x in (0.1, -0.1, 0.05, -0.05, 0.0, 0.1, -0.1, 0.02): b8.add(x)
        self.assertLess(abs(b1.z(1.0)), abs(b8.z(1.0)) + 1e-9)            # few answers -> less certain -> smaller |z| for the same deviation... or equal
        self.assertAlmostEqual(F.PersonBaseline(5, 1).z(5.0), 0.0, places=6)

    def test_stouffer_unit_variance_under_correlation(self):
        rho, k = 0.5, 4
        cov = np.full((k, k), rho) + np.eye(k) * (1 - rho)
        z = np.random.default_rng(0).multivariate_normal(np.zeros(k), cov, 40000)
        m = F.FusionModel(rho={"arousal": rho})
        names = ["f0_mean_st", "f0_sd_st", "ser_arousal", "face_activity"]
        Z = np.array([m.group_stat(dict(zip(names, row)), "arousal")[0] for row in z[:8000]])
        self.assertAlmostEqual(float(Z.var()), 1.0, delta=0.06)                # corrected statistic is ~N(0,1) under the null
        naive = np.array([np.sum(row) / np.sqrt(k) for row in z[:8000]])
        self.assertGreater(float(naive.var()), 1.5)                            # ignoring correlation inflates evidence ~2x

    def test_llr_formula_and_additive_attribution(self):
        m = F.FusionModel()
        zs = {"response_latency_s": 2.0, "f0_mean_st": 1.0, "detail_density": -2.0}
        r = m.score(zs)
        for g in ("load", "arousal", "content"):
            Z = r.group_z[g]; self.assertAlmostEqual(r.group_llr[g], round(m.d[g] * Z - m.d[g] ** 2 / 2, 4), places=3)
        total_cue = sum(c.llr for c in r.contributions)
        self.assertAlmostEqual(total_cue, sum(m.d[g] * r.group_z[g] for g in r.group_z), places=2)   # exact additive attribution

    def test_nonverbal_alone_never_flags(self):
        m = F.FusionModel()
        zs = {"response_latency_s": 3, "pause_rate_per_min": 3, "f0_mean_st": 3, "ser_arousal": 3, "self_adaptor_rate_per_min": 3, "face_activity": 3}
        r = m.score(zs)
        self.assertTrue(r.nonverbal_capped)
        self.assertEqual(r.flag, "none")
        self.assertLess(r.log_odds, F._logit(0.5))

    def test_content_hard_evidence_flags(self):
        r = F.FusionModel().score({}, {"nli_contradiction": 1.0})
        self.assertEqual(r.flag, "verify"); self.assertIn("nli_contradiction", r.hard_evidence)

    def test_missing_modalities_abstain(self):
        r = F.FusionModel().score({}, {}, {"voice": False, "vision": False, "text": False})
        self.assertFalse(r.usable); self.assertEqual(r.flag, "none")

    def test_fit_recovers_effect_and_never_flips_sign(self):
        rng = np.random.default_rng(3)
        rows = [({"detail_density": -(rng.normal(0.8 * y, 1)) * -1 * -1}, y) for y in (rng.random(400) < 0.4).astype(int)]
        m = F.FusionModel(); rep = m.fit(rows)
        self.assertIn("content", rep); self.assertGreaterEqual(m.d["content"], 0.0)
        rows0 = [({"detail_density": rng.normal()}, int(rng.random() < .4)) for _ in range(300)]
        m0 = F.FusionModel(); m0.fit(rows0); self.assertLess(m0.d["content"], 0.6)   # no effect -> stays near/below prior scale

    def test_conformal_coverage_guarantee_on_exchangeable_data(self):
        rng = np.random.default_rng(5)
        def draw(n):
            y = (rng.random(n) < 0.4).astype(int); p = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + 1.5 * (y - 0.5)))); return p, y
        pc, yc = draw(600); pt, yt = draw(6000)
        g = F.ConformalGate(0.1).fit(pc, yc); self.assertTrue(g.valid)
        for cls, name in ((0, "supported"), (1, "unsupported")):
            cov = np.mean([name in g.predict_set(p) for p, y in zip(pt, yt) if y == cls])
            self.assertGreaterEqual(cov, 0.9 - 0.03)

    def test_conformal_refuses_with_tiny_calibration(self):
        g = F.ConformalGate(0.1).fit(np.random.rand(10), np.array([0, 1] * 5))
        self.assertFalse(g.valid); self.assertEqual(g.decide(0.9)[0], "uncalibrated")


class TestEval(unittest.TestCase):
    def test_grouped_folds_never_split_a_subject(self):
        subj = [f"s{i // 5}" for i in range(100)]
        for tr, te in E.grouped_folds(subj, 5):
            self.assertFalse({subj[i] for i in tr} & {subj[i] for i in te})

    def test_auc_known_values(self):
        self.assertEqual(E.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(E.auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)

    def test_null_data_is_at_chance(self):
        rows = S.simulate(50, 10, shift={}, seed=11)
        r = E.evaluate(rows, B=200); lo, hi = r["auc_ci95"]
        self.assertTrue(lo <= 0.5 <= hi, (lo, hi))

    def test_signal_is_detected_and_content_dominates_when_built_in(self):
        sh = {"response_latency_s": .5, "f0_mean_st": .5, "detail_density": .9, "claim_detail_gap": .9}
        rows = S.simulate(50, 10, shift=sh, seed=12)
        r = E.evaluate(rows, B=200); self.assertGreater(r["auc_ci95"][0], 0.6)
        ab = E.ablation(rows); self.assertGreater(ab["content_only"], ab["nonverbal_only"])

    def test_heldout_conformal_coverage_close_to_target(self):
        sh = {"response_latency_s": .6, "f0_mean_st": .6, "detail_density": .9, "claim_detail_gap": .9, "ser_arousal": .5}
        g = E.calibrated_gate(S.simulate(60, 10, shift=sh, seed=13))
        for k in ("coverage_class0", "coverage_class1"):
            self.assertGreaterEqual(g[k], 0.82)


class TestText(unittest.TestCase):
    def test_detail_density_and_filler(self):
        rich = T.text_cues("Um, I built it in PyTorch for two years, we deployed it in 2023 and it crashed because of a leak, so I profiled it and cut latency by 40%.", ["pytorch"])
        poor = T.text_cues("I did some work on it and it was good and I learned many things from the project overall.")
        self.assertGreater(rich.detail_density, poor.detail_density * 3); self.assertGreater(rich.filler_rate_per_100w, 0)

    def test_quantity_conflict(self):
        self.assertEqual(T.quantity_conflict("5 years of Python experience", "I used python for six months")[0], 1.0)
        self.assertEqual(T.quantity_conflict("2 years of Python", "about 18 months of python")[0], 0.0)

    def test_whole_term_matching(self):
        self.assertFalse(T.has_term("I made a classifier", "c")); self.assertTrue(T.has_term("I know C, Python", "c")); self.assertFalse(T.has_term("C#", "c"))

    def test_nli_pair_filtering_and_hard_evidence(self):
        nli = T.NLIChecker(predict=lambda pairs: np.array([[0.9, 0.05, 0.05]] * len(pairs)))
        p, shown = T.contradiction_evidence(["I deployed YOLO to production for two years."], "I only tried YOLO once in a class.", ["YOLO"], nli)
        self.assertGreater(p, 0.85); self.assertTrue(shown)
        p2, _ = T.contradiction_evidence(["I deployed YOLO to production."], "I like football very much indeed.", ["YOLO"], nli)
        self.assertEqual(p2, 0.0)                                              # different topic -> no NLI call


class TestVoice(unittest.TestCase):
    sr = 16000

    def _voiced(self, dur, f0, amp=.3):
        t = np.arange(int(dur * self.sr)) / self.sr
        x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8)); return (amp * x / np.max(np.abs(x))).astype(np.float32)

    def test_latency_from_tts_clock_pause_pitch(self):
        rng = np.random.default_rng(0); sil = lambda d: rng.normal(0, .002, int(d * self.sr)).astype(np.float32)
        a = np.concatenate([sil(1.6), self._voiced(2.0, 150), sil(0.5), self._voiced(2.5, 150), sil(.3)])
        v = V.voice_cues(a, self.sr, capture_t0=100.0, tts_end=99.6, n_words=14)
        self.assertAlmostEqual(v.response_latency_s, 2.0, delta=0.1); self.assertEqual(v.n_pauses, 1)
        self.assertAlmostEqual(v.f0_mean_st, 12 * np.log2(1.5), delta=0.4); self.assertTrue(v.usable)

    def test_barge_in_not_scored(self):
        a = np.concatenate([self._voiced(4.0, 150)]).astype(np.float32)
        v = V.voice_cues(a, self.sr, capture_t0=100.0, tts_end=101.5, n_words=10)
        self.assertTrue(v.barge_in); self.assertIsNone(v.fusion_cues()["response_latency_s"])

    def test_short_or_noisy_audio_is_unusable(self):
        rng = np.random.default_rng(1); a = rng.normal(0, .01, self.sr * 5).astype(np.float32)
        v = V.voice_cues(a, self.sr, 0.0, None, n_words=5); self.assertFalse(v.usable)
        self.assertIsNone(v.fusion_cues()["f0_mean_st"])

    def test_second_voice_drift_larger_for_different_source(self):
        same = np.concatenate([self._voiced(4, 120), self._voiced(4, 120)]); diff = np.concatenate([self._voiced(4, 120), self._voiced(4, 230)])
        ds = V.second_voice_drift(same, self.sr, V.energy_vad(same, self.sr)); dd = V.second_voice_drift(diff, self.sr, V.energy_vad(diff, self.sr))
        self.assertGreater(dd, ds + 0.02)

    def test_ser_adapter(self):
        c = V.ser_to_cues({"model1_dimensional": {"arousal": .7}, "model2_categorical": {"fear_cluster_score": .4}, "model3_hubert": {"stress_score": .2}})
        self.assertAlmostEqual(c["ser_arousal"], .7); self.assertAlmostEqual(c["ser_negative_mass"], .3)


class TestVision(unittest.TestCase):
    fs, Tn = 30, 20

    def _mk(self, dx, dy=None, yaw=None, pit=None):
        t = np.arange(0, self.Tn, 1 / self.fs); n = len(t); z = np.zeros(n)
        return [NS(t=t[i], face_ok=True, gaze_valid=True, gaze_dx=dx[i], gaze_dy=(z if dy is None else dy)[i], yaw_dev=(z if yaw is None else yaw)[i],
                   pitch_dev=(z if pit is None else pit)[i], looking=abs(dx[i]) < .5) for i in range(n)]

    def test_reading_both_directions_vs_free_gaze(self):
        rng = np.random.default_rng(1); t = np.arange(0, self.Tn, 1 / self.fs)
        ph = (t % 2.5) / 2.5; saw = np.where(ph < .92, ph / .92, 1 - (ph - .92) / .08) * 1.2 - .6
        for sgn in (1, -1):
            self.assertGreaterEqual(X.vision_extras(self._mk(sgn * saw + rng.normal(0, .03, len(t)), rng.normal(0, .05, len(t)))).reading_score, C.READING_SCORE_FLAG)
        free = np.clip(np.cumsum(rng.normal(0, .06, len(t))), -.9, .9); free = free - free.mean()
        for g in (60, 200, 350, 500): free[g:g + 20] += .9
        self.assertLess(X.vision_extras(self._mk(free + rng.normal(0, .05, len(t)))).reading_score, C.READING_SCORE_FLAG)

    def test_offscreen_fixation_and_nods(self):
        t = np.arange(0, self.Tn, 1 / self.fs); fix = np.zeros(len(t)); fix[100:300] = 1.1
        self.assertAlmostEqual(X.vision_extras(self._mk(fix)).offscreen_fixation_s, 6.6, delta=0.3)
        nod = 4 * np.sin(2 * np.pi * 1.2 * t) * (t > 5) * (t < 12)
        r = X.vision_extras(self._mk(np.zeros(len(t)), pit=nod)); self.assertGreater(r.head_nod_rate_per_min, 15); self.assertEqual(r.head_shake_rate_per_min, 0)

    def test_shake_from_absolute_yaw_is_not_lost(self):
        t = np.arange(0, self.Tn, 1 / self.fs); shake = 4 * np.sin(2 * np.pi * 1.2 * t) * (t > 5) * (t < 12)
        r = X.vision_extras(self._mk(np.zeros(len(t)), yaw=np.abs(shake)))               # Vision v3 gives |yaw - baseline|
        self.assertGreater(r.head_shake_rate_per_min, 12); self.assertTrue(any("absolute" in n for n in r.notes))

    def test_offscreen_uses_vision_tolerance_units(self):
        t = np.arange(0, self.Tn, 1 / self.fs); g = np.zeros(len(t)); g[100:300] = 1.2        # >= 1.0 tolerance units = "away" in classify_gaze_direction
        self.assertGreater(X.vision_extras(self._mk(g)).offscreen_fixation_s, 6.0)
        g2 = np.zeros(len(t)); g2[100:300] = 0.7                                                # inside tolerance = still looking at the screen
        self.assertEqual(X.vision_extras(self._mk(g2)).offscreen_fixation_s, 0.0)

    def test_too_few_valid_frames(self):
        s = self._mk(np.zeros(self.fs * self.Tn)); [setattr(x, "face_ok", False) for x in s[::2]]
        self.assertFalse(X.vision_extras(s).usable)


class TestClaimsAndTTS(unittest.TestCase):
    def test_rule_claims_and_grounding(self):
        R = "Built a pose app in PyTorch for 2 years and deployed it to production serving 5k users. Reduced latency by 40%."
        cs = CE.rule_claims(R, ["pytorch"]); self.assertTrue(cs and cs[0].months == 24 and cs[0].production == "yes")
        fake = [CE.Claim59("kubernetes", "Ran Kubernetes clusters", "resume")]
        self.assertEqual(CE.grounded(fake + cs, R), cs)

    def test_nuextract_styles_with_stub_server(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        seen = []
        class H(BaseHTTPRequestHandler):
            def do_POST(s):
                body = json.loads(s.rfile.read(int(s.headers["Content-Length"]))); seen.append((s.path, body))
                out = {"claims": [{"skill_or_topic": "PyTorch", "quote": "Built a pose app in PyTorch for 2 years", "duration": "2 years"}]}
                txt = "x " + json.dumps(out); r = {"content": txt} if s.path == "/completion" else {"choices": [{"message": {"content": txt}}]}
                b = json.dumps(r).encode(); s.send_response(200); s.send_header("Content-Length", str(len(b))); s.end_headers(); s.wfile.write(b)
            def log_message(s, *a): pass
        srv = HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            for style in ("1.5", "2.0"):
                r = CE.NuExtractClient(f"http://127.0.0.1:{srv.server_address[1]}", style).extract("Built a pose app in PyTorch for 2 years and more.")
                self.assertEqual(r[0].months, 24)
            self.assertIn("<|input|>", seen[0][1]["prompt"]); self.assertIn("# Template:", seen[1][1]["messages"][0]["content"])
        finally:
            srv.shutdown()

    def test_timed_tts_records_end(self):
        class SD:
            def play(s, a, sr, device=None): time.sleep(.2)
            def wait(s): pass
        class K:
            def create(s, text, voice, speed, lang): return [0.0] * 10, 24000
        k = TT.KokoroTTS(kokoro_factory=K, sd_module=SD()); self.assertTrue(k.initialize())
        t = TT.TimedTTS(k); t0 = time.monotonic(); end = t.say("hello"); k.close()
        self.assertGreaterEqual(end - t0, 0.19); self.assertIsNone(TT.TimedTTS(None).say("x"))


class TestBridgeIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import test_v58 as T58, sim_candidate_v58 as SIM
        except Exception as e:                                  # V58 files not on the path
            raise unittest.SkipTest(f"V58 package not importable: {e}")
        cls.T58, cls.SIM = T58, SIM

    def _run(self, nli):
        import interviewer_bridge_v59 as B
        r, j = self.T58.resume(), self.T58.job()
        sess = B.BehaviorSession(self.T58.RESUME_TXT, [], nli=nli)
        eng = B.BehaviorAwareEngine(r, j, None, session=sess)
        out = eng.start(); n = 0; srcs = []
        while True:
            out = eng.step(self.SIM.answer(eng.ctx, out.text), B.Signals()); n += 1
            srcs.append(eng.last_decision.source if eng.last_decision else "")
            if out.finished or n > 30: break
        return eng, sess, srcs, B

    def test_no_signal_no_probe_and_interview_unchanged(self):
        eng, sess, srcs, B = self._run(None)
        self.assertEqual(sess.probes_used, 0); self.assertNotIn("behavior_probe", srcs); self.assertEqual(sess.errors, [])
        rep = B.behavior_report(eng); self.assertEqual(rep["summary"]["answers"], len(sess.turns))

    def test_probe_policy_caps_and_outcomes(self):
        nli = T.NLIChecker(predict=lambda pairs: np.array([[0.95, 0.02, 0.03]] * len(pairs)))
        eng, sess, srcs, B = self._run(nli)
        self.assertGreaterEqual(sess.probes_used, 1); self.assertLessEqual(sess.probes_used, C.MAX_PROBES_PER_INTERVIEW)
        self.assertEqual(len(sess.probed_topics), sess.probes_used)                                   # one probe per topic
        self.assertEqual(sess.errors, [])
        first_probe_turn = [i + 1 for i, s in enumerate(srcs) if s == "behavior_probe"][0]
        self.assertGreaterEqual(first_probe_turn + 0, C.MIN_TURN_FOR_PROBE - 1)
        self.assertTrue(all(p["status"] in ("verified", "unresolved", "inconsistent") for p in sess.probe_log))
        texts = [t.interviewer for t in eng.transcript]; self.assertTrue(any("least expected" in x or "concrete moment" in x or "which part did you" in x or "how long" in x for x in texts))

    def test_bridge_failure_never_stops_the_interview(self):
        import interviewer_bridge_v59 as B
        r, j = self.T58.resume(), self.T58.job(); sess = B.BehaviorSession("", [])
        sess.analyze = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        eng = B.BehaviorAwareEngine(r, j, None, session=sess); out = eng.start()
        out = eng.step(self.SIM.answer(eng.ctx, out.text), None)
        self.assertTrue(out.text); self.assertTrue(any("boom" in e for e in sess.errors))


if __name__ == "__main__":
    unittest.main()


class TestScaling(unittest.TestCase):
    def test_registry_loads_once_under_contention(self):
        import scaling_v59 as SC
        reg = SC.ModelRegistry(); made = []
        def factory(): time.sleep(0.05); made.append(1); return object()
        res = []
        ts = [threading.Thread(target=lambda: res.append(reg.get("nli", factory))) for _ in range(16)]
        [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(len(made), 1); self.assertEqual(len({id(r) for r in res}), 1); self.assertEqual(reg.loads["nli"], 1)

    def test_pool_timeout_returns_fallback_and_overload_degrades(self):
        import scaling_v59 as SC
        p = SC.AnalysisPool(workers=1, max_pending=1)
        self.assertEqual(p.run(lambda: (time.sleep(0.3), "slow")[1], timeout=0.05, fallback="fb"), "fb"); self.assertEqual(p.timeouts, 1)
        time.sleep(0.4)
        self.assertEqual(p.run(lambda: "fast", timeout=1.0, fallback="fb"), "fast")
        p.shutdown()

    def test_atomic_report_and_eviction(self):
        import scaling_v59 as SC, tempfile
        with tempfile.TemporaryDirectory() as d:
            s = SC.SessionStore(d, idle_ttl_s=10)
            s.create("a", lambda: object()); path = s.save_report("a", {"x": 1})
            self.assertEqual(json.loads(path.read_text())["x"], 1)
            self.assertEqual(s.evict_idle(now=time.time() + 100), ["a"])
            with self.assertRaises(KeyError):
                s.create("b", lambda: object()); s.create("b", lambda: object())


class TestConcurrentInterviews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import test_v58 as T58, sim_candidate_v58 as SIM
        except Exception as e:
            raise unittest.SkipTest(f"V58 package not importable: {e}")
        cls.T58, cls.SIM = T58, SIM

    def test_24_parallel_sessions_are_isolated_and_match_serial_runs(self):
        import scaling_v59 as SC, interviewer_bridge_v59 as B
        reg = SC.ModelRegistry()
        def nli():
            return reg.get("nli", lambda: T.NLIChecker(predict=lambda pairs: np.array([[0.9, 0.05, 0.05]] * len(pairs))))
        def factory(params):
            sess = B.BehaviorSession(self.T58.RESUME_TXT, [], nli=nli())
            return B.BehaviorAwareEngine(self.T58.resume(), self.T58.job(), None, session=sess)
        def run_one(svc, sid):
            out = svc.start(sid, {}); n = 0
            while not out.finished and n < 30:
                out = svc.step(sid, self.SIM.answer(svc.store.entry(sid).engine.ctx, out.text), B.Signals()); n += 1
            rep, _ = svc.finish(sid, B.behavior_report)
            return (rep["summary"]["answers"], rep["summary"]["probes"], tuple(p["status"] for p in rep["probes"]))
        serial = SC.BehaviorService(factory, registry=reg); ref = run_one(serial, "ref")
        svc = SC.BehaviorService(factory, registry=reg); results, errors = {}, []
        def worker(i):
            try: results[i] = run_one(svc, f"s{i}")
            except Exception as e: errors.append(repr(e))
        ts = [threading.Thread(target=worker, args=(i,)) for i in range(24)]
        [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(errors, []); self.assertEqual(len(results), 24)
        self.assertTrue(all(r == ref for r in results.values()), set(results.values()) | {ref})     # same input -> same behaviour, no cross-talk
        self.assertEqual(reg.loads["nli"], 1); self.assertEqual(svc.store.ids(), [])
