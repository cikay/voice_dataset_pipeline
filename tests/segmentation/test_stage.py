import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from voice_dataset_pipeline.segmentation.stage import (
    SegmentationStage,
    normalize_text,
    split_into_sentences,
)


@pytest.fixture
def stage():
    return SegmentationStage(
        {
            "input_manifest": "dataset/enhancement_manifest.json",
            "output_manifest": "dataset/segmentation_manifest.json",
            "audio_dir": "dataset/audio_segments",
            "align_language": "kmr",
        }
    )


class TestNormalizeText:

    def test_basic_normalization(self):
        assert normalize_text("   Silav hevalno   ") == "Silav hevalno"

    def test_single_spaces_preserved(self):
        assert normalize_text("Ramyar diçû") == "Ramyar diçû"

    def test_tabs_and_newlines_normalized(self):
        assert normalize_text("Nasa\t\tdîsa\ndiçe heyvê") == "Nasa dîsa diçe heyvê"

    def test_mixed_whitespace(self):
        assert (
            normalize_text("  ez  \n\t  diçim  \r  dibistanê  ") == "ez diçim dibistanê"
        )

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_whitespace_only_string(self):
        assert normalize_text("   \t\n  ") == ""

    @pytest.mark.parametrize(
        "text_decomposed,expected_normalized",
        [
            ("Ez dixwazim bi kurdî biaxivim.", "Ez dixwazim bi kurd\xee biaxivim."),
            (
                "Rojbaş, em ê herin şaredariyê.",
                "Rojbaş, em \xea herin şaredariy\xea.",
            ),
        ],
    )
    def test_unicode_normalization(self, text_decomposed, expected_normalized):
        assert text_decomposed != expected_normalized
        result = normalize_text(text_decomposed)
        assert result == expected_normalized
        assert isinstance(result, str)
        assert len(result) > 0

    def test_no_modification_needed(self):
        assert normalize_text("Ez dikim biçim mala xwe") == "Ez dikim biçim mala xwe"


class TestSplitIntoSentences:

    def test_single_sentence_period(self):
        assert split_into_sentences("Ew diçû mektebê.") == ["Ew diçû mektebê."]

    def test_single_sentence_exclamation(self):
        assert split_into_sentences("Ev pir xweş e!") == ["Ev pir xweş e!"]

    def test_single_sentence_question(self):
        assert split_into_sentences("Tu kî yî?") == ["Tu kî yî?"]

    def test_multiple_sentences(self):
        result = split_into_sentences("Ramyar diçû derve. Êvara we bixêr! Tu çawa yî?")
        assert result == ["Ramyar diçû derve.", "Êvara we bixêr!", "Tu çawa yî?"]

    def test_multiple_punctuation_marks(self):
        result = split_into_sentences("Wow!! Rastî?? Belê!!!")
        assert result == ["Wow!!", "Rastî??", "Belê!!!"]

    def test_empty_string(self):
        assert split_into_sentences("") == []

    def test_no_punctuation(self):
        assert split_into_sentences("Ez ditirsim te nebînim") == []

    def test_leading_trailing_whitespace(self):
        result = split_into_sentences("   Yekem.   Duyem.   ")
        assert result == ["Yekem.", "Duyem."]
        for sentence in result:
            assert sentence == sentence.strip()

    def test_multiple_spaces_between_sentences(self):
        assert len(split_into_sentences("Yekem.    Duyem!")) == 2

    def test_sentence_with_special_characters(self):
        result = split_into_sentences("Salav (cîhan)! Nivîsa baş [nûçe]?")
        assert result == ["Salav (cîhan)!", "Nivîsa baş [nûçe]?"]

    def test_mixed_punctuation(self):
        assert len(split_into_sentences("Dest pê bike! Lawîn tê. Ev kî ye?")) == 3

    def test_abbreviation(self):
        result = split_into_sentences("Prof. Dr. Smith diçe fezayê.")
        assert result == ["Prof.", "Dr.", "Smith diçe fezayê."]
