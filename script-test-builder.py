import pandas as pd
from pathlib import Path

# Dados de teste
blocks = [
    ("poissons", [
        "3_4_poissons_eagleray_3310",
        "3_5_poissons_hammerhead_3495",
        "3_3_poissons_tigershark_3244"
    ]),
    ("chiens", [
        "1_2_chiens_boxer_1146",
        "1_4_chiens_goldenretriever_1423",
        "1_5_chiens_Rottweiler_1578"
    ]),
    ("singes", [
        "4_3_singes_squirrelmonkey_4082",
        "4_2_singes_gorilla_4004",
        "4_1_singes_chimpanzee_3772"
    ])
]

rows = []
for classe, filenames in blocks:
    for name in filenames:
        # Reconstroi subclasse a partir do padrão
        parts = name.split("_")
        subclasse = parts[3]  # Ex: 'eagleray'
        full_path = Path("MIR_DATASETS_B/MIR_DATASETS_B") / classe / subclasse / f"{name}.jpg"
        rows.append({
            "query_image_path": str(full_path),
            "expected_class": classe
        })

df = pd.DataFrame(rows)
df.to_csv("test_queries.csv", index=False)
print("✅ test_queries.csv generated.")
