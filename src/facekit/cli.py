"""FaceKit CLI entry point."""
import typer

from facekit.commands.average_face import average_face
from facekit.commands.extract_features import extract_features
from facekit.commands.extract_features_custom import extract_features_custom
from facekit.commands.extract_landmarks import extract_landmarks
from facekit.commands.generate import generate
from facekit.commands.pack import pack
from facekit.commands.resolve_diseases import resolve_diseases
from facekit.commands.score import score

app = typer.Typer(
    help="FaceKit: rare disease facial phenotype analysis toolkit.",
    no_args_is_help=True,
)

# Module 1: Landmark extraction
app.command("extract-landmarks")(extract_landmarks)

# Module 2: Average face generation
app.command("average-face")(average_face)

# Module 3: Geometric phenotype features
app.command("extract-features")(extract_features)

# Module 3a: Custom user mapping variant (no MONDO; user JSON + optional plugin file)
app.command("extract-features-custom")(extract_features_custom)

# Module 3b: Disease-name resolver helper (MONDO cache builder)
app.command("resolve-diseases")(resolve_diseases)

# Module 3c: Normative z-scoring against a control reference
app.command("score")(score)

# Module 4: Synthetic face generation (StyleGAN3)
app.command("pack")(pack)
app.command("generate")(generate)

# Future commands will be registered here:
# app.command("enhance")(enhance)     # Module 4: DDColor + GFPGAN preprocessing
# app.command("train")(train)         # Module 4: train a generator
# app.command("privacy")(privacy)     # Module 5: identity / appearance leakage audit


if __name__ == "__main__":
    app()