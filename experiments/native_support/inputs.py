"""Only observed tokens and their prompt boundary enter detection."""

from state_audit.storage import read_json


def load_responses(path):
    manifest = read_json(path)
    responses = []
    for row in manifest["responses"]:
        tokens = row["token_ids"]
        prompt = row["prompt_length"]
        if not 0 < prompt < len(tokens):
            raise ValueError(f"{row['id']}: expected prompt followed by observed answer tokens")
        responses.append({
            "id": row["id"], "source_id": row["source_id"], "token_ids": tokens,
            "prompt_length": prompt, "token_text": row["token_text"],
        })
    if len({row["id"] for row in responses}) != len(responses):
        raise ValueError("Response IDs must be unique")
    return manifest["model"], responses


def validate_tokenizer(response, tokenizer):
    tokens, pieces = response["token_ids"], response["token_text"]
    if len(tokens) != len(pieces):
        raise ValueError(f"{response['id']}: token ID/text lengths differ")
    for position, (token, piece) in enumerate(zip(tokens, pieces)):
        if tokenizer.decode([token]) != piece:
            raise ValueError(f"{response['id']}: tokenizer mismatch at position {position}")
