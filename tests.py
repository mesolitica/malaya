import unittest
import malaya

class TestMethods(unittest.TestCase):
    def test_malaya(self):
        self.assertEqual(malaya.to_ordinal(11), 'kesebelas')

if __name__ == '__main__':
    unittest.main()
