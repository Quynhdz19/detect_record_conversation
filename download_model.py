"""Download the Vietnamese Zipformer ASR used for live captions."""

from app.asr import MODEL_ID, get_zipformer


def main() -> None:
    print(f"Downloading/loading {MODEL_ID} ...")
    get_zipformer()
    print("Done. Model cached and ready.")


if __name__ == "__main__":
    main()
