import argparse, csv, sys

# GOALS = [
#     {
#         "id": "G1_biodiversity_vs_land_use",
#         "label": "How land-use form & intensity affect biodiversity",
#         "text": "how the form and intensities of land use affect biodiversity"
#     },
#     {
#         "id": "G2_ecosystem_process_vs_land_use",
#         "label": "How land-use form & intensity affect ecosystem processes",
#         "text": "how the form and intensities of land use affect ecosystem processes"
#     },
#     {
#         "id": "G3_biodiversity_interactions",
#         "label": "How different components of biodiversity interact",
#         "text": "how different components of biodiversity interact"
#     },
#     {
#         "id": "G4_biodiversity_to_services",
#         "label": "How biodiversity influences ecosystem processes & services",
#         "text": "how different components of biodiversity influence ecosystem processes and ecosystem services"
#     }
# ]

GOALS = [
    {
        "id": "G1_biodiversity_vs_land_use",
        "label": "How land-use form & intensity affect biodiversity",
        "text": """Investigate how land-use configuration and management intensity shape species richness and community composition.

Quantify biodiversity responses along gradients of land-use type and use intensity.

Determine how different land-use practices and their intensity alter the diversity of taxa and functional groups.

Assess the effects of landscape structure and land-use pressure on biodiversity patterns.

Examine how the form and degree of land use drive changes in organisms and their diversity."""
    },
    {
        "id": "G2_ecosystem_process_vs_land_use",
        "label": "How land-use form & intensity affect ecosystem processes",
        "text": """Evaluate how land-use types and their intensity influence key ecosystem processes such as productivity and nutrient cycling.

Test how management regime and land-use configuration modify rates of decomposition, carbon fluxes, and energy flow.

Analyse process-level responses (e.g., soil respiration, primary production) along land-use intensity gradients.

Determine how spatial arrangement and intensity of land use affect ecosystem functioning.

Examine the sensitivity of ecosystem processes to differences in land-use form and management intensity."""
    },
    {
        "id": "G3_biodiversity_interactions",
        "label": "How different components of biodiversity interact",
        "text": """Explore interactions among biodiversity components (genes, species, functional traits, trophic levels) within ecosystems.

Investigate how species, functional groups, and trophic guilds influence each other through competition, facilitation, and predation.

Determine the network structure of biodiversity interactions and how they vary across contexts.

Analyse cross-component linkages (taxonomic, functional, phylogenetic) and their ecological consequences.

Assess how changes in one biodiversity facet propagate to others through biotic interactions."""
    },
    {
        "id": "G4_biodiversity_to_services",
        "label": "How biodiversity influences ecosystem processes & services",
        "text": """Quantify how biodiversity drives ecosystem functioning and the delivery of services such as pollination, carbon storage, and water regulation.

Test the relationships between biodiversity (richness, evenness, trait diversity) and ecosystem process rates and service provision.

Determine how losses or gains in biodiversity alter ecosystem processes and the benefits people receive.

Evaluate the pathways by which biodiversity affects ecosystem functioning and service outcomes.

Examine biodiversity–function relationships and their implications for ecosystem services across landscapes."""
    }
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","label","text"])
        w.writeheader()
        for g in GOALS:
            w.writerow(g)
    print(f"Wrote {args.out} with {len(GOALS)} goals.", file=sys.stderr)

if __name__ == "__main__":
    main()
