import time
import mysql
import mysql.connector
import requests
import logging
from typing import List, Dict
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
URL_INSERT = "http://118.70.52.237:9920/insert"
URL_SEARCH = "http://localhost:9920/search"
CHECKPOINT_FILE = "checkpoint4.txt"

def insert_url(url: str, metadata: Dict[str, str]) -> bool:
    """Insert a URL with specific metadata into the service."""
    payload = {
        "url": url, 
        "metadata": metadata
    }
    try:
        response = requests.post(URL_INSERT, json=payload, timeout = 15)
        response.raise_for_status()
        logging.info(f"Inserted {url} successfully.")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to insert {url} : {str(e)}")
        return False

def search_url(url: str, k: int) -> None:
    """Search for a URL with a limit of results."""
    payload = {"url": url, "k": k}
    try:
        response = requests.post(URL_SEARCH, json=payload)
        response.raise_for_status()
        logging.info(f"Search successful for {url}. Results: {response.json()}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to search {url}: {str(e)}")

# def batch_insert(urls_and_metadata: List[Dict[str, Dict[str, str]]]) -> None:
#     """Insert multiple URLs with their respective metadata."""
#     for entry in urls_and_metadata:
#         url = entry["url"]
#         metadata = entry.get("metadata", {})
#         insert_url(url, metadata)

def batch_search(urls: List[str], k: int = 10) -> None:
    """Search multiple URLs with a defined limit of results."""
    for url in urls:
        search_url(url, k)

def get_urls_from_db():
    try:
        # Kết nối đến MySQL database
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="password02",
            database="kc"
        )
        cursor = connection.cursor()

        # Truy vấn các URL từ bảng (thay thế table_name và url_column bằng tên bảng và cột URL của bạn)
        query = """
        SELECT urls.page_id, urls.source, raw_page.title, SUBSTRING_INDEX(content, '==', 1) as abstract
        FROM raw_page
        JOIN urls ON raw_page.id = urls.page_id 
        """
        cursor.execute(query)

        # Lấy tất cả các URL từ bảng
        src_list = []
        tittle_list = []
        abstract_list = []
        for row in cursor.fetchall():
            src_list.append(row[1])
            tittle_list.append(row[2])
            abstract_list.append(row[3])

        # Đóng kết nối
        cursor.close()
        connection.close()

        return src_list, tittle_list, abstract_list
    except mysql.connector.Error as err:
        logging.error(f"Database error: {err}")
        return [], []

def read_checkpoint() -> int:
    """Đọc chỉ số cuối cùng đã xử lý từ file checkpoint."""
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0  # Nếu không có checkpoint, bắt đầu từ 0

def update_checkpoint(last_index: int):
    """Cập nhật checkpoint với chỉ số mới nhất."""
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(last_index))
    
if __name__ == "__main__":
    
    src_list, tittle_list, abstract_list = get_urls_from_db()
    
    # Sample data with individual metadata for each URL   
    IMAGE_DATA = [
        {"url": url, "metadata": {"tittle": tittle, "abstract": abstract}}
        for url, tittle, abstract in zip(src_list, tittle_list, abstract_list)
    ]

    logging.info("Starting batch insertion...")
    total = len(IMAGE_DATA)

    start_time = time.time()
    last_index = read_checkpoint()
    data_to_insert = IMAGE_DATA[last_index:]
    
    # Use tqdm progress bar in the main function to monitor the process
    for idx, entry in enumerate(tqdm(data_to_insert, desc="Inserting URLs", total=len(data_to_insert), unit="url"), start=last_index):
        url = entry["url"]
        metadata = entry.get("metadata", {})

        # Insert the URL and update checkpoint if successful
        if insert_url(url, metadata):
            update_checkpoint(idx + 1)

    end_time = time.time()
    logging.info(f"Batch insertion completed in {(end_time - start_time):.2f} seconds")

