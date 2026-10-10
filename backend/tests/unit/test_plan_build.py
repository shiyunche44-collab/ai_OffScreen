"""Tests for the incremental-plan decisions, text similarity and vector signal (all pure)."""

from __future__ import annotations

import numpy as np
import pytest

from offscreen.algo.plan_build import (
    Decision,
    PlanReport,
    decide_narration,
    keep_locked,
    plan_order,
    playback_ms,
    text_digest,
    voice_for,
    with_fresh_footage,
)
from offscreen.algo.textsim import bigram_similarity, bigrams, coverage
from offscreen.algo.vectors import rank_by_similarity, search_signal
from offscreen.domain.plan import AudioRef, Clip, EditPlan, PlanSegment, ScriptRef, VoiceSpec

VOICE = VoiceSpec(voice_id="v", speed=1.0)


def clip(a: int, b: int, *, locked: bool = False, speed: float = 1.0, shot: str = "sh_1") -> Clip:
    return Clip(
        asset_id="ast_1", shot_id=shot, src_in_ms=a, src_out_ms=b, speed=speed, locked=locked
    )


def narration(text: str, *, stale: bool = False, voice: VoiceSpec = VOICE) -> PlanSegment:
    return PlanSegment(
        id="seg_01",
        kind="narration",
        text=text,
        text_hash=text_digest(text),
        stale=stale,
        voice=voice,
        audio=AudioRef(file="tts/a.wav", duration_ms=1000),
        clips=[clip(0, 1000)],
    )


class TestDecide:
    def test_unchanged_segment_is_reused(self) -> None:
        d = decide_narration("seg_01", "你好", VOICE, narration("你好"), audio_exists=True)

        assert d == Decision("seg_01", None) and d.reuse

    @pytest.mark.parametrize(
        ("text", "previous", "voice", "audio", "reason"),
        [
            ("你好", None, VOICE, True, "new"),
            ("你好啊", narration("你好"), VOICE, True, "text"),
            ("你好", narration("你好", stale=True), VOICE, True, "stale"),
            ("你好", narration("你好", voice=VoiceSpec(voice_id="w")), VOICE, True, "voice"),
            ("你好", narration("你好"), VoiceSpec(voice_id="v", speed=1.2), True, "voice"),
            ("你好", narration("你好"), VOICE, False, "audio_missing"),
        ],
    )
    def test_reasons(
        self, text: str, previous: PlanSegment | None, voice: VoiceSpec, audio: bool, reason: str
    ) -> None:
        d = decide_narration("seg_01", text, voice, previous, audio_exists=audio)

        assert d.reason == reason and not d.reuse

    def test_a_previous_segment_without_a_hash_is_rebuilt(self) -> None:
        old = narration("你好").model_copy(update={"text_hash": None})

        assert (
            decide_narration("seg_01", "你好", VOICE, old, audio_exists=True).reason == "unhashed"
        )

    def test_an_original_segment_of_that_id_does_not_count_as_narration(self) -> None:
        old = PlanSegment(id="seg_01", kind="original", clips=[clip(0, 500, locked=True)])

        assert decide_narration("seg_01", "你好", VOICE, old, audio_exists=True).reason == "new"


class TestLockedClips:
    def test_nothing_locked_means_everything_is_still_to_fill(self) -> None:
        kept = keep_locked([clip(0, 1000), clip(1000, 2000)], 5000)

        assert (kept.clips, kept.insert_at, kept.remaining_ms) == ([], 0, 5000)

    def test_new_footage_takes_the_place_of_the_first_unlocked_clip(self) -> None:
        a, b, c = clip(0, 1000), clip(5000, 6500, locked=True), clip(9000, 10_000, locked=True)
        kept = keep_locked([a, b, c], 6000)

        fresh = [clip(20_000, 23_000, shot="sh_9")]
        assert kept.insert_at == 0 and kept.remaining_ms == 6000 - 1500 - 1000
        assert with_fresh_footage(kept, fresh) == [*fresh, b, c]

    def test_locked_clips_before_the_gap_stay_in_front(self) -> None:
        a, b, c = clip(0, 1000, locked=True), clip(2000, 3000), clip(4000, 5000, locked=True)
        kept = keep_locked([a, b, c], 4000)
        fresh = [clip(9000, 11_000, shot="sh_9")]

        assert with_fresh_footage(kept, fresh) == [a, *fresh, c]

    def test_all_locked_adds_footage_at_the_end(self) -> None:
        a, b = clip(0, 1000, locked=True), clip(2000, 3000, locked=True)
        kept = keep_locked([a, b], 5000)

        assert kept.insert_at == 2 and kept.remaining_ms == 3000

    def test_locked_footage_longer_than_the_voice_over_leaves_nothing_to_fill(self) -> None:
        kept = keep_locked([clip(0, 4000, locked=True)], 3000)

        assert kept.remaining_ms == -1000

    def test_speed_counts(self) -> None:
        assert playback_ms(clip(0, 2000, speed=0.8)) == 2500


def test_report_lists_what_was_done() -> None:
    r = PlanReport(
        rebuilt={"seg_02": "text"}, reused=["seg_01"], removed=["seg_09"], warnings=["overlap"]
    )

    assert r.lines() == [
        "plan: 1 reused, 1 rebuilt",
        "plan: rebuilt seg_02 (text)",
        "plan: reused seg_01",
        "plan: removed seg_09 (no longer in the script)",
        "plan: warning: overlap",
    ]


class TestTextSimilarity:
    def test_bigrams_ignore_spaces_and_punctuation(self) -> None:
        assert bigrams("龙，飞 了！") == {"龙飞", "飞了"}
        assert bigrams("A") == {"a"} and bigrams("  。") == set()

    def test_chinese_without_spaces_still_compares(self) -> None:
        assert bigram_similarity("一条龙在天上飞", "一条龙在天上飞") == 1.0
        assert bigram_similarity("一条龙在天上飞", "一只猫在地上跑") < 0.2
        assert bigram_similarity("", "龙") == 0.0

    def test_coverage_is_the_share_of_the_description_in_the_text(self) -> None:
        assert coverage("龙在天上飞", "女孩看见一条龙在天上飞过") == 1.0
        assert coverage("龙在天上飞", "女孩回家了") == 0.0
        assert coverage("", "任何") == 0.0


class TestVectors:
    UNIT = np.eye(3, dtype=np.float32)

    def test_ranks_every_shot_by_cosine(self) -> None:
        ranked = rank_by_similarity(["a", "b", "c"], self.UNIT, [0.1, 0.9, 0.3])

        assert [s for s, _ in ranked] == ["b", "c", "a"]
        assert ranked[0][1] == pytest.approx(0.9 / np.linalg.norm([0.1, 0.9, 0.3]))

    def test_ties_keep_shot_order(self) -> None:
        ranked = rank_by_similarity(["a", "b", "c"], self.UNIT, [1, 1, 1])

        assert [s for s, _ in ranked] == ["a", "b", "c"]

    def test_mismatches_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="2 shots but 3 vectors"):
            rank_by_similarity(["a", "b"], self.UNIT, [1, 0, 0])
        with pytest.raises(ValueError, match="zero"):
            rank_by_similarity(["a", "b", "c"], self.UNIT, [0, 0, 0])

    def test_signal_stretches_each_column_and_averages(self) -> None:
        image = [("a", 0.30), ("b", 0.20), ("c", 0.10)]
        text = [("b", 0.90), ("a", 0.50), ("c", 0.10)]

        top, scores = search_signal({"image": image, "text": text}, top_k=2)

        assert scores == {"a": pytest.approx(0.75), "b": pytest.approx(0.75), "c": 0.0}
        assert set(top) == {"a", "b"}

    def test_a_single_flat_column_scores_zero(self) -> None:
        _, scores = search_signal({"text": [("a", 0.5), ("b", 0.5)]})

        assert scores == {"a": 0.0, "b": 0.0}

    def test_no_columns_no_signal(self) -> None:
        assert search_signal({}) == ([], {})


class TestVoiceAndOrder:
    def seg(self, sid: str, kind: str = "narration") -> PlanSegment:
        if kind == "original":
            return PlanSegment(id=sid, kind="original", clips=[clip(0, 500, locked=True)])
        return narration("你好").model_copy(update={"id": sid})

    def plan(self, *segs: PlanSegment) -> EditPlan:
        return EditPlan(
            id="pln_t",
            project_id="prj_t",
            version=3,
            author="human",
            script_ref=ScriptRef(id="scr_t", version=2),
            segments=list(segs),
        )

    def test_a_pinned_voice_outlives_the_scripts(self) -> None:
        mine = VoiceSpec(voice_id="mine", speed=1.3)
        pinned = narration("你好", voice=mine).model_copy(update={"voice_pinned": True})

        assert voice_for(pinned, VOICE) == mine
        assert voice_for(narration("你好", voice=mine), VOICE) == VOICE  # not pinned
        assert voice_for(None, VOICE) == VOICE
        assert decide_narration(
            "seg_01", "你好", voice_for(pinned, VOICE), pinned, audio_exists=True
        ).reuse

    def test_without_history_the_plan_is_the_script(self) -> None:
        assert plan_order(["a", "b"], None, None) == ["a", "b"]
        assert plan_order(["a", "b"], self.plan(self.seg("seg_02")), None) == ["a", "b"]

    def test_a_person_s_order_stands(self) -> None:
        prev = self.plan(self.seg("seg_03"), self.seg("seg_01"), self.seg("seg_02"))

        order = plan_order(["seg_01", "seg_02", "seg_03"], prev, {"seg_01", "seg_02", "seg_03"})

        assert order == ["seg_03", "seg_01", "seg_02"]

    def test_a_segment_the_person_deleted_stays_deleted(self) -> None:
        prev = self.plan(self.seg("seg_01"), self.seg("seg_03"))

        order = plan_order(["seg_01", "seg_02", "seg_03"], prev, {"seg_01", "seg_02", "seg_03"})

        assert order == ["seg_01", "seg_03"]

    def test_a_segment_the_script_dropped_goes_and_a_new_one_follows_its_predecessor(self) -> None:
        prev = self.plan(self.seg("seg_03"), self.seg("seg_01"), self.seg("seg_02"))

        order = plan_order(
            ["seg_01", "seg_04", "seg_03"],  # seg_02 dropped, seg_04 new after seg_01
            prev,
            {"seg_01", "seg_02", "seg_03"},
        )

        assert order == ["seg_03", "seg_01", "seg_04"]

    def test_a_new_first_segment_goes_first_and_inserted_original_sound_stays(self) -> None:
        prev = self.plan(self.seg("seg_01"), self.seg("seg_o01", "original"), self.seg("seg_02"))

        order = plan_order(["seg_00", "seg_01", "seg_02"], prev, {"seg_01", "seg_02"})

        assert order == ["seg_00", "seg_01", "seg_o01", "seg_02"]

    def test_an_original_segment_the_script_dropped_is_not_mistaken_for_an_insert(self) -> None:
        prev = self.plan(self.seg("seg_01"), self.seg("seg_02", "original"))

        assert plan_order(["seg_01"], prev, {"seg_01", "seg_02"}) == ["seg_01"]
