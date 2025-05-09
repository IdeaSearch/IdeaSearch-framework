from unittest import TestCase
from unittest import main as unittest_main
from ansatz import check_ansatz_format
from ansatz_testcases import check_format_testcases


class TestCheckAnsatzFormat(TestCase):
    
    def test_check_ansatz_format(self):
        for i, case in enumerate(check_format_testcases):
            with self.subTest(i=i, expr=case.expression):
                result = check_ansatz_format(case.expression, case.variables, case.functions)
                self.assertEqual(result, case.expected, msg=f"用例 {i} 失败: {case.expression}")


if __name__ == "__main__":
    unittest_main()
