import hopsworks
import os
from dotenv import load_dotenv
import sys
load_dotenv()


def main():
    api_key = os.getenv("HOPSWORKS_API_KEY")

    if not api_key:
        print("❌ HOPSWORKS_API_KEY not found.")
        sys.exit(1)

    print("🔐 Logging into Hopsworks...")
    project = hopsworks.login(api_key_value=api_key)

    print(f"✅ Logged into project: {project.name}")

    mr = project.get_model_registry()

    print("\n📦 Fetching models from registry...\n")

    # Try wildcard fetch
    try:
        models = mr.get_models(name="1_year_champ")  # Empty string to list all
    except Exception as e:
        print("❌ Failed to fetch models:", e)
        return

    if not models:
        print("❌ No models found.")
        return

    print(f"🔢 Total Models Found: {len(models)}\n")

    for m in models:
        print(f"Name: {m.name}")
        print(f"Version: {m.version}")
        print(f"Training Metrics: {getattr(m, 'training_metrics', {})}")
        print("-" * 60)

if __name__ == "__main__":
    main()