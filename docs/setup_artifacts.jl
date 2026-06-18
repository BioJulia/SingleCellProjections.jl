# You only need to run this if the artifacts have changed.
# To rerun, first delete Artifacts.toml and then run this file.

using ArtifactUtils

for name in ["mutation_status", "scRNA_AML", "AML9", "AML7", "AML21", "AML10", "AML25", "AML24", "AML23", "AML32", "AML28", "AML27", "AML34", "AML33", "AML61", "AML55", "AML48", "AML37", "AML83", "AML80", "AML79", "AML62", "AML97", "AML85R", "AML85D", "AML117", "AML111", "AML110", "AML105", "AML104", "AML126", "AML124", "AML123", "AML151", "AML138", "AML136", "AML161", "AML157", "AML155", "NBM8-CD34", "NBM7-MNC", "NBM4-MNC", "AML172", "NBM10-CD34", "NBM8-MNC", "NBM11-MNC", "NBM11-CD34", "NBM10-MNC"]

	add_artifact!(
		"docs/Artifacts.toml",
		"Lilljebjorn2025_$(name)",
		"https://github.com/rasmushenningsson/SingleCellExampleData/releases/download/Lilljebjorn2025/$(name).tar.gz";
		lazy = true)
end