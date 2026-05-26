from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from voice_dataset_pipeline.filter.stage import (
    FilterStage,
    MIN_DURATION,
    MAX_DURATION,
    MIN_SCORE,
)


@pytest.fixture
def stage():
    return FilterStage(
        {
            "input_manifest": "dataset/segmentation_manifest.json",
            "output_manifest": "dataset/filter_manifest.json",
        }
    )


class TestShouldDiscard:

    def test_digit_detection(self, stage: FilterStage):
        result, reason = stage._should_discard("Sala 2024 pir zehmet bû", 5.0, 0.0)
        assert result is True
        assert reason == "digits"

    def test_exclude_number_can_be_disabled(self):
        stage = FilterStage(
            {
                "input_manifest": "dataset/segmentation_manifest.json",
                "output_manifest": "dataset/filter_manifest.json",
                "exclude_number": False,
            }
        )
        result, reason = stage._should_discard("Sala 2024 pir zehmet bû", 5.0, 0.0)
        assert result is False
        assert reason == ""

    def test_abbreviation(self, stage: FilterStage):
        result, reason = stage._should_discard(
            "PDK û PYD du partîyên siyasî yên Kurdan ne.", 5.0, 0.0
        )
        assert result is True
        assert reason == "abbreviations"

    def test_exclude_abbr_can_be_disabled(self):
        stage = FilterStage(
            {
                "input_manifest": "dataset/segmentation_manifest.json",
                "output_manifest": "dataset/filter_manifest.json",
                "exclude_abbr": False,
            }
        )
        result, reason = stage._should_discard(
            "PDK û PYD du partîyên siyasî yên Kurdan ne.", 5.0, 0.0
        )
        assert result is False
        assert reason == ""

    def test_dots_abbreviation(self, stage: FilterStage):
        result, reason = stage._should_discard("D.Y.A navê welatekî ye", 5.0, 0.0)
        assert result is True
        assert reason == "abbreviations"

    @pytest.mark.parametrize(
        "sentence",
        [
            "Ocalan bê berdan, bê meclisê û li wir banga bidawîbûna PKKê bike, dewlet jî ji bo kurdan nasîna mafên bingehîn misoger bike",
            "YPG/YPJ rêxistinên çekdarî yên kurd in",
            "li başûrê Kurdistanê bi hevkariya PDKê xwest",
            "AKP-MHP demokrasîyê naxwazin",
            "desthilatdîya AKP-MHPê demokrasîyê naxwazin",
        ],
    )
    def test_abbreviation_with_suffix(self, sentence, stage: FilterStage):
        result, reason = stage._should_discard(sentence, 5.0, 0.0)
        assert result is True
        assert reason == "abbreviations"

    def test_duration_too_short(self, stage: FilterStage):
        result, reason = stage._should_discard(
            "Ev hevok rast e.", MIN_DURATION - 0.1, 0.0
        )
        assert result is True
        assert reason == "duration"

    def test_duration_too_long(self, stage: FilterStage):
        result, reason = stage._should_discard(
            "Ev hevok rast e.", MAX_DURATION + 0.1, 0.0
        )
        assert result is True
        assert reason == "duration"

    def test_duration_at_min_boundary(self, stage: FilterStage):
        result, _ = stage._should_discard("Ev hevok baş e.", MIN_DURATION, 0.0)
        assert result is False

    def test_duration_at_max_boundary(self, stage: FilterStage):
        result, _ = stage._should_discard("Ev hevok baş e.", MAX_DURATION, 0.0)
        assert result is False

    def test_too_few_words(self, stage: FilterStage):
        sentence = "ez têm"
        result, reason = stage._should_discard(sentence, 5.0, 0.0)
        assert result is True
        assert reason == "too_few_words"

    def test_min_words_boundary(self, stage: FilterStage):
        sentence = "ez têm malê"
        result, _ = stage._should_discard(sentence, 5.0, 0.0)
        assert result is False

    def test_low_align_score(self, stage: FilterStage):
        result, reason = stage._should_discard("Ev hevok baş e.", 5.0, MIN_SCORE - 0.1)
        assert result is True
        assert reason == "low_score"

    def test_align_score_at_boundary(self, stage: FilterStage):
        result, _ = stage._should_discard("Ev hevok şaş e.", 5.0, MIN_SCORE)
        assert result is False


class TestFilterEntries:

    def test_filters_entries_and_keeps_original_payload(self, stage: FilterStage):
        entries = [
            {
                "id": "keep",
                "text": "Ev hevok baş e.",
                "duration": 5.0,
                "align_score": 0.0,
                "word_count": 4,
                "audio_file": "keep.wav",
            },
            {
                "id": "drop",
                "text": "Sala 2024 pir zehmet bû",
                "duration": 5.0,
                "align_score": 0.0,
                "word_count": 5,
                "audio_file": "drop.wav",
            },
        ]

        kept, discard_counts = stage._filter_entries(entries)

        assert kept == [entries[0]]
        assert discard_counts == {"digits": 1}
