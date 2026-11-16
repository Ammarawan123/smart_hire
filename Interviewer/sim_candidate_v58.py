"""sim_candidate_v58.py — scripted persona that answers like the V51 candidate (short, broken English, honest about gaps).
Used for the automated regression run (--simulate) and the V51-vs-V58 comparison. Answers depend on WHAT is asked, not on turn number."""
from __future__ import annotations

import re

from models_v58 import QuestionContext

P = {
    "intro": "I am BSAI student who likes to learn new things and this role is of SOftware Developer which i always wants to be and this also matches my skillset",
    "fraday": "It is a desktop assistant. It listen my voice and convert to text, then send to Gemini API and Cohere for the answer and it speak back with pyttsx3. I made the whole thing myself in Python.",
    "sign": "It is like it does the detection of sign gestures in real time and converts then in text and i myself built it whole",
    "own": "I wrote the python code myself and trained the model. The YOLO and OpenCV libraries are from other people, I only use them.",
    "dataset": "I collect it myself like i download the dataset of some images and if a class which i want i cant find then i take my image of making sign gestures",
    "labeling": "I used Roboflow to draw boxes on every image, around 1500 images and 26 classes. I check the labels again because some were wrong.",
    "training": "I used pretrained yolov8n and train for 50 epochs with image size 640. I choose small model because my laptop is slow.",
    "evaluation": "I checked mAP on the validation set, it was around 0.85. And I test with my webcam also.",
    "challenge": "Some letters looks same so accuracy was low. I add more images for these letters and train again, then it is better.",
    "prompt": "I just write the prompt in Python like you are an assistant, answer short. Sometimes the answer was very long so I add a word limit in the prompt and it was better.",
    "rag_no": "I haven't used RAG",
    "rag_scn": "I think we put the documents in a vector database, then search the similar parts for the question and give these parts to the LLM in the prompt so it answer from them.",
    "gate": "I know basic C# but not ASP.NET or Angular. I used SQL in university.",
    "sql": "I join the two tables on customer id, then group by customer and sum the order value, order by that sum descending and take top 5.",
    "oop": "OOP is classes and objects, inheritance and polymorphism. In Pygame I made a class for player and one for enemy.",
    "rest": "REST API is when client send HTTP request like GET or POST to the server and server reply with JSON. I use Gemini API like this in Fraday.",
    "sdlc": "First requirements, then design, coding, testing, deployment and maintenance.",
    "arch": "I dont know much about it",
    "practices": "I use Git for my projects and I write comments in the code.",
    "team": "In group project if the project is multi modal so we divide one will do one part and other will do different and one is for the presentation",
    "team2": "My part was the model. Once the other part was late so I wrote small test code with fake data so I can continue, and later we joined it.",
    "solve": "In Fraday the speech was not working in the start. I check the microphone code step by step and found the wrong sample rate, then I fixed it.",
    "learn": "I learned YOLO from YouTube and documentation in about two weeks and then made the project.",
    "default": "I am not very sure, I think I need to learn more about this.",
    "closing": "No",
}


def answer(ctx: QuestionContext, question: str) -> str:
    q = question.lower()
    t = ctx.targets
    k = ctx.kind
    if k == "intro":
        return P["intro"]
    if k == "closing":
        return P["closing"]
    if k == "contradiction":
        return "I only learned it, I did not use it in a project."
    if k == "project_walkthrough":
        return P["fraday"] if "fraday" in (ctx.anchor_project or "").lower() else P["sign"] if "sign" in (ctx.anchor_project or "").lower() else "It is a small project, I made it with Python and a classifier."
    if ctx.anchor_project and not ctx.aspect and t:
        key0 = t[0]
        if key0 == "oop":
            return P["oop"]
        if key0 in ("rest_api",):
            return P["rest"]
        if key0 in ("prompt_eng", "llm", "genai"):
            return P["prompt"]
        if key0 == "ml_libs":
            return "I used YOLOv8 from Ultralytics. I load a pretrained model, give my dataset in a yaml file, train and then check mAP on validation images."
    if ctx.anchor_project:
        a = ctx.aspect
        if a == "ownership":
            return P["own"]
        if a in P:
            return P[a]
        if "prompt" in q:
            return P["prompt"]
        if "rest" in q or "request and the response" in q:
            return P["rest"]
        return P["challenge"] if "hardest" in q or "problem" in q else P["own"]
    if k == "behavioural":
        s = ctx.soft
        return {"teamwork": P["team2"] if "someone else" in q else P["team"], "problem_solving": P["solve"], "learning_ability": P["learn"]}.get(s, P["solve"])
    key = t[0] if t else ""
    if k == "cluster_gate":
        return P["gate"]
    if key == "rag":
        return P["rag_scn"] if ctx.scenario else P["rag_no"]
    if key in ("prompt_eng", "llm"):
        return P["prompt"]
    if key in ("sqlserver", "sql"):
        return P["sql"]
    if key == "oop":
        return P["oop"]
    if key == "rest_api":
        return P["rest"]
    if key == "sdlc":
        return P["sdlc"]
    if key == "architecture":
        return P["arch"]
    if key == "se_practices":
        return P["practices"]
    if key == "csharp":
        return "I only wrote small console programs in C# for university, like a calculator and a student list."
    if key in ("python", "ml_libs"):
        return "I use Keras to make the CNN, I split the data in train and test, train it and check the accuracy. In Python I use lists and dictionary a lot."
    if key == "data_analysis":
        return "I use pandas to clean the data and matplotlib to plot, I look for missing values and outliers first."
    return P["default"]
