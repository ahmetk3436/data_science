import csv
import requests
from bs4 import BeautifulSoup


def get_soup(url):
    r = requests.get(url)
    return BeautifulSoup(r.content, "html.parser")


def extract_table_data(soup):
    table = soup.find("div", {"id": "content"}).find("table", {"id": "sortierbare_tabelle"})
    if table:
        return table.find_all("tr")
    return []


def write_to_csv(file, headers, rows):
    writer = csv.writer(file, delimiter=' ')
    writer.writerow(headers)
    for i, row in enumerate(rows):
        data = []
        for cell in row.find_all("td", recursive=False):
            if "Archived" in cell.text.strip():
                break
            data.append(cell.text.strip())
        writer.writerow(data)
        print(data)


def main():
    url = "https://www.notebookcheck.net/Mobile-Processors-Benchmark-List.2436.0.html"
    soup = get_soup(url)

    with open('mobile_cpu.csv', 'w', encoding='utf-8', newline='') as file:
        headers = ['Pos', 'Model', 'Codename', 'L2 Cache + L3 Cache', 'TDP Watt', 'MHz - Turbo', 'Cores / Threads',
                   'Perf. Rating', 'Cinebench R15 CPU Single 64Bit', 'Cinebench R15 CPU Multi 64Bit',
                   'Cinebench R23 Single Core', 'Cinebench R23 Multi Core', 'x265', 'Blender(-)', '7-Zip Single',
                   '7-Zip', 'Geekbench 5.5 Single-Core', 'Geekbench 5.5 Multi-Core', 'WebXPRT 3']

        rows = extract_table_data(soup)

        if not rows:
            print("Tablo içeriği bulunamadı.")
            return

        write_to_csv(file, headers, rows)


if __name__ == "__main__":
    main()
