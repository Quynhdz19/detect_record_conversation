"""Download PhoWhisper-small into local HuggingFace cache."""

from app.asr import MODEL_ID, get_transcriber, pick_device


def main() -> None:
    device, dtype = pick_device()
    print(f"Downloading/loading {MODEL_ID} on {device}/{dtype} ...")
    get_transcriber()
    print("Done. Model cached and ready.")


if __name__ == "__main__":
    main()
