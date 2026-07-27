from unsloth import FastLanguageModel
import json

max_seq_length = 2048

# Point this at your finished adapter folder.
MODEL_PATH = "outputs_v2/final"

SYSTEM_PROMPT = (
    "You are an expert quantitative trading assistant specialized in market "
    "structure analysis, breakout detection, and trap identification."
)

# Add/edit tickers and dates you want to spot-check here.
TEST_CASES = [
    {"ticker": "AAPL", "date": "2026-02-03"},
    {"ticker": "MSFT", "date": "2026-01-15"},
    {"ticker": "NVDA", "date": "2026-02-10"},
    {"ticker": "TSLA", "date": "2026-01-28"},
]

REQUIRED_FIELDS = [
    "ticker",
    "breakout_date",
    "entry_date",
    "entry_price",
    "outcome",
    "5_day_sequence",
]


def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate(model, tokenizer, ticker, date):
    prompt = (
        f"Analyze the technical sequence and breakout structure for "
        f"{ticker} on {date}."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    )
    text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return prompt, text


def validate(raw_text):
    """Check the model's output is parseable JSON with the expected fields."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}", None

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return False, f"Missing fields: {missing}", data

    seq = data.get("5_day_sequence")
    if not isinstance(seq, list) or len(seq) == 0:
        return False, "5_day_sequence is empty or not a list", data

    return True, "OK", data


def main():
    model, tokenizer = load_model()

    results = []
    for case in TEST_CASES:
        prompt, raw = generate(model, tokenizer, case["ticker"], case["date"])
        ok, msg, parsed = validate(raw)
        results.append({
            "ticker": case["ticker"],
            "date": case["date"],
            "valid": ok,
            "message": msg,
            "raw_output": raw,
        })

        print("=" * 80)
        print(f"PROMPT: {prompt}")
        print(f"VALID: {ok} ({msg})")
        print("RAW OUTPUT:")
        print(raw)
        print()

    passed = sum(1 for r in results if r["valid"])
    print("=" * 80)
    print(f"SUMMARY: {passed}/{len(results)} outputs were valid JSON matching schema")

    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Full results saved to test_results.json")


if __name__ == "__main__":
    main()
