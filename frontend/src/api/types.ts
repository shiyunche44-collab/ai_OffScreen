// Names for the generated schema types (src/api/schema.d.ts, produced by `make api-types`).
// Nothing here describes backend data by hand (ARCHITECTURE R7): every type is looked up in the
// generated schema.
import type { components, paths } from "./schema";

export type { paths };
export type Schemas = components["schemas"];
export type Job = Schemas["Job"];
export type JobStatus = Job["status"];
export type MediaAsset = Schemas["MediaAsset"];
export type AssetDetail = Schemas["AssetDetail"];
export type Project = Schemas["Project"];
export type ProjectDetail = Schemas["ProjectDetail"];
export type ErrorBody = Schemas["ErrorBody"];
export type ProjectOptions = Schemas["ProjectOptions"];
export type Script = Schemas["Script"];
export type ScriptSegment = Script["segments"][number];
export type ShotsView = Schemas["ShotsView"];
export type ShotView = ShotsView["shots"][number];
export type Transcript = Schemas["Transcript"];
export type TranscriptLine = Transcript["lines"][number];
export type Scenes = Schemas["Scenes"];
export type Scene = Scenes["scenes"][number];
export type Story = Schemas["Story"];
export type CharactersView = Schemas["CharactersView"];
export type Character = CharactersView["characters"][number];
export type CharacterEdit = Schemas["CharacterEdit"];
export type CutsView = Schemas["CutsView"];
export type CutEvaluation = Schemas["CutEvaluation"];
export type OutlineBeat = Schemas["OutlineBeat"];
export type OutlineView = Schemas["OutlineView"];
export type DocumentVersion = Schemas["DocumentVersion"];
export type DocumentDiff = Schemas["DocumentDiff"];
export type StylePreset = Schemas["StylePreset"];
