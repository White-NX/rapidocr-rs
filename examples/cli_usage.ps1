param(
    [string]$Image = "$env:RAPIDOCR_PYTHON_REPO\python\tests\test_files\ch_en_num.jpg",
    [string]$ModelDir = "models"
)

# Default registered model set. Missing model assets are downloaded into $ModelDir.
cargo run -p rapidocr-cli -- --image $Image --model-dir $ModelDir

# Generate a reusable TOML config for the default model set.
cargo run -p rapidocr-cli -- --model-set ppocrv6-small --write-default-config config\ppocrv6-small.toml --model-dir $ModelDir

# Run from a pre-populated model directory without network access.
cargo run -p rapidocr-cli -- --image $Image --config config\ppocrv6-small.toml --no-download

# Recognition-only smoke with a non-default English model set.
$EnglishImage = "$env:RAPIDOCR_PYTHON_REPO\python\tests\test_files\en_rec.jpg"
cargo run -p rapidocr-cli -- --model-set ppocrv5-en-mobile --image $EnglishImage --model-dir $ModelDir --no-det --no-cls
