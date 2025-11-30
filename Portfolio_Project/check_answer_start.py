import json

with open("cpw_addendum.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for article in data["data"]:
    for para in article["paragraphs"]:
        context = para["context"]
        for qa in para["qas"]:
            for ans in qa["answers"]:
                text = ans["text"]
                start = ans["answer_start"]
                extracted = context[start:start+len(text)]
                if extracted != text:
                    print(f"Mismatch in {qa['id']}:")
                    print(f"  expected: {repr(text)}")
                    print(f"  got     : {repr(extracted)}")
