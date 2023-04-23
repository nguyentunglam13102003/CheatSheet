binary_form_1 = str(input(""))
binary_form_2 = str(input(""))

def fitlen(str_1, str_2):
    if (len(str_1) < len(str_2)):
        str_1 = '0' * (len(str_2) - len(str_1)) + str_1
    else:
        str_2 = '0' * (len(str_1) - len(str_2)) + str_2
    return str_1, str_2

def add(str_1, str_2):
    if len(str_1) != len(str_2):
        str_1, str_2 = fitlen(str_1, str_2)

    res = ""
    carry = 0
    for i in range(len(str_1) - 1, -1, -1):
        a = int(str_1[i])
        b = int(str_2[i])
        val = a ^ b ^ carry
        res = str(val) + res
        carry = (a & b) | (a & carry) | (b & carry)
    if carry:
        res = '1' + res
    return res

def karatsuba(str_1, str_2):
   if len(str_1) != len(str_2):
       str_1, str_2 = fitlen(str_1, str_2)

   n = len(str_1)

   if n == 0:
       return 0
   if n == 1:
       return int(str_1[0])*int(str_2[0])

   first_half = n//2
   second_half = n - first_half

   left_1 = str_1[:first_half]
   right_1 = str_1[first_half:]

   left_2 = str_2[:first_half]
   right_2 = str_2[first_half:]

   res1 = karatsuba(left_1, left_2)
   res2 = karatsuba(right_1, right_2)
   res3 = karatsuba(add(left_1, right_1), add(left_2, right_2))

   return res1*(1<<(2*second_half)) + (res3 - res1 - res2) * (1 << second_half) + res2

print(karatsuba(binary_form_1, binary_form_2))