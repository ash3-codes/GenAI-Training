# test_rag.py

from rag.orchestrator import HRPolicyRAG

rag = HRPolicyRAG()

response = rag.ask("How many sick leaves are allowed per year?")

print("Answer:\n")
print(response["answer"])

print("\nSources:")
for s in response["sources"]:
    print(s)