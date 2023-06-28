import requests

if True:  # these work.
    url = "https://www.ebi.ac.uk/proteins/api/proteins"
    params = {
        "offset": "0",
        "size": "10000",
        "reviewed": "true",
        "organism": "Bacteriophage",
        "format": "fasta"
    }

    response = requests.get(url, params=params)

    if response.ok:
        filename = "bacteriophage.fasta"
        with open(filename, "w") as f:
            f.write(response.text)
        print(f"FASTA sequences written to {filename}")
    else:
        print(
            f"Error downloading sequences: {response.status_code} {response.reason}")

    params = {
        "offset": "0",
        "size": "100",
        "reviewed": "true",
        "format": "csv",
        "organism": "Bacteriophage",
        "columns": "id,entry name,protein names,genes,length,organism"
    }

    response = requests.get(url, params=params)

    if response.ok:
        filename = "bacteriophage.txt"
        with open(filename, "w") as f:
            f.write(response.text)
        print(f"Protein information written to {filename}")
    else:
        print(
            f"Error downloading protein information: {response.status_code} {response.reason}")