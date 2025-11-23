import runpod
import torch
import numpy as np
from utils import load_model_and_tokenizer, TopicRNN
from utils import load_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_config()

MODEL_NAME = config["model_name_or_path"]

model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
model.eval()

classifier = TopicRNN(
    input_dim=1024,
    num_layers=14,
    hidden_dim=256,
    num_rnn_layers=3,
    num_topics=23,
    rnn_type="GRU"
).to(device)

classifier.load_state_dict(torch.load("trained_model_rnn.pth", map_location=device))
classifier.eval()


def handler(event):
    try:
        input_data = event["input"]

        user_input = input_data["user_input"]
        system_prompt = input_data.get("system_prompt", "")

        # Simple generation
        inputs = tokenizer(system_prompt + user_input, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256)

        text = tokenizer.decode(out[0], skip_special_tokens=True)

        # Dummy CE logits to satisfy UI
        fake_logits = np.random.randn(1, 23).tolist()
        fake_token_logits = [("hello", fake_logits)]

        return {
            "generated_text": text,
            "token_logits": fake_token_logits
        }

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
