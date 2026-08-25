from glean_gepa.adapter_types import (
    SingleModelALDataInst,
    SingleModelALRolloutOutput,
    TeacherStudentALDataInst,
    TeacherStudentALRolloutOutput,
)


def test_single_model_types_only_include_shell_error_evidence():
    data_keys = SingleModelALDataInst.__required_keys__ | SingleModelALDataInst.__optional_keys__
    output_keys = SingleModelALRolloutOutput.__required_keys__ | SingleModelALRolloutOutput.__optional_keys__

    assert {"eval_entry_id", "eval_run_id", "eval_trace_id"} <= data_keys
    assert {"shell_error_messages", "shell_action_inputs", "eval_trace_id"} <= output_keys
    assert "teacher_answer" not in output_keys
    assert "teacher_tool_events" not in output_keys


def test_teacher_student_types_only_include_paired_execution_evidence():
    data_keys = TeacherStudentALDataInst.__required_keys__ | TeacherStudentALDataInst.__optional_keys__
    output_keys = TeacherStudentALRolloutOutput.__required_keys__ | TeacherStudentALRolloutOutput.__optional_keys__

    assert "eval_trace_id" not in data_keys
    assert {"student_answer", "student_tool_events", "teacher_answer", "teacher_tool_events"} <= output_keys
    assert "shell_error_messages" not in output_keys
    assert "shell_action_inputs" not in output_keys
