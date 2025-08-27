import re
# 示例文本，包含多个银行卡号
text = "您的银行卡号是：622188123456789011，请妥善保管。另外，请核对您的卡号：4234567890123456，是否无误。"
# 编写正则表达式，匹配常见的银行卡号
# 注意：这里仅作为示例，实际中银行卡号规则可能更复杂
pattern = r'\b(?:[456]\d{15}|[2-9]\d{17})\b'
# 使用findall方法查找所有匹配项
card_numbers = re.findall(pattern, text)
print(card_numbers)

def luhn_check(card_number):
    def digits_of(n):
        return [int(d) for d in str(n)]
    digits = digits_of(card_number[:-1])  # 去掉最后一位（校验位）
    odd_digits = digits[-1::-2]  # 奇数位置的数字
    even_digits = digits[-2::-2]  # 偶数位置的数字
    checksum = 0
    checksum += sum(odd_digits)
    for digit in even_digits:
        checksum += sum(digits_of(digit*2))
    return checksum % 10 == 0
# 测试Luhn算法
card_numbers = ['62176888888', '4234567890123456']
for card in card_numbers:
    print(f"Card '{card}' is valid: {luhn_check(card)}")