import os
import matplotlib.pyplot as plt


def read_spark_output(path):
    """Read key-value results saved by Spark."""
    result = {}

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            key, value = line.split(":", 1)

            result[key.strip()] = float(value.strip())

    return result


os.makedirs("img", exist_ok=True)

# Quality distribution

quality_data = read_spark_output(
    "output/quality_distribution/part-00000"
)

qualities = list(quality_data.keys())
counts = list(quality_data.values())

plt.figure(figsize=(10, 6))

plt.bar(qualities, counts)

plt.xlabel("Wine quality")
plt.ylabel("Number of wines")
plt.title("Distribution of Wine Quality Scores")

plt.tight_layout()

plt.savefig(
    "img/quality_distribution.png",
    dpi=300
)

plt.close()


# Average alcohol by type

type_data = read_spark_output(
    "output/type_analysis/part-00000"
)

wine_types = list(type_data.keys())
alcohol_values = list(type_data.values())

plt.figure(figsize=(8, 6))

plt.bar(wine_types, alcohol_values)

plt.xlabel("Wine type")
plt.ylabel("Average alcohol content")
plt.title("Average Alcohol Content by Wine Type")

plt.tight_layout()

plt.savefig(
    "img/alcohol_by_type.png",
    dpi=300
)

plt.close()

print("Charts successfully saved to img/")