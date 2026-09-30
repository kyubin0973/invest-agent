import threading
import unittest

from rag.retriever import get_vectorstore, search_industry, search_technology


class RetrieverConcurrencyTests(unittest.TestCase):
    def test_parallel_first_search_shares_one_vectorstore(self):
        """Technology ∥ Market처럼 여러 스레드가 동시에 처음 검색해도 충돌 없이 같은 벡터DB를 쓴다."""
        errors, stores = [], []

        def work(i: int) -> None:
            try:
                stores.append(id(get_vectorstore()))
                if i % 2:
                    search_technology("real-world deployment", "Figure AI", k=2)
                else:
                    search_industry("humanoid market size", k=2)
            except Exception as e:  # noqa: BLE001 - 스레드 안의 예외를 모아 검사한다
                errors.append(f"{type(e).__name__}: {e}")

        threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(set(stores)), 1)


if __name__ == "__main__":
    unittest.main()
