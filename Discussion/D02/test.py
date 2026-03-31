import importlib.metadata
import pprint

# Get the full mapping of all installed packages
mapping = importlib.metadata.packages_distributions()

# Pretty print the result to see all mappings
pprint.pprint(mapping)