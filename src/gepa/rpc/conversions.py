"""Conversions between protobuf messages and Python objects passed to gepa."""

from __future__ import annotations

from dataclasses import dataclass

from gepa.rpc.generated import gepa_pb2 as pb


@dataclass
class RemoteExample:
    id: str
    fields: dict[str, str]

    def to_proto(self) -> pb.Example:
        return pb.Example(id=self.id, fields=dict(self.fields))

    @classmethod
    def from_proto(cls, msg: pb.Example) -> RemoteExample:
        return cls(id=msg.id, fields=dict(msg.fields))


@dataclass
class RemoteTrajectory:
    input_fields: dict[str, str]
    output: str
    feedback: str

    def to_proto(self) -> pb.Trajectory:
        return pb.Trajectory(
            input_fields=dict(self.input_fields),
            output=self.output,
            feedback=self.feedback,
        )

    @classmethod
    def from_proto(cls, msg: pb.Trajectory) -> RemoteTrajectory:
        return cls(
            input_fields=dict(msg.input_fields),
            output=msg.output,
            feedback=msg.feedback,
        )


def reflective_data_to_python(
    data: pb.ReflectiveDatasetResponse,
) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {}
    for component, payload in data.reflective_data.items():
        records: list[dict[str, object]] = []
        for entry in payload.entries:
            records.append(
                {
                    "Inputs": dict(entry.inputs),
                    "Generated Outputs": entry.generated_output,
                    "Feedback": entry.feedback,
                }
            )
        out[component] = records
    return out
