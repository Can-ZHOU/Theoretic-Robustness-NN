x0 = 0.21132513590986978
width = x0 - 0.20960408399359702

# x1_list = [0.209688, 0.20973, 0.209751, 0.209761, 0.209766, 0.209769, 0.20977]
# x1_list = [0.209723, 0.209783, 0.209813, 0.209828, 0.209835]
x1_list = [0.20972329035847216, 0.20978289611897948, 0.20981269964330992, 0.20982760156643924, 0.20983505256823804, 0.20983877807919512, 0.20984064083718797]

for i in range(len(x1_list)):
    ratio = (x0-x1_list[i]) / width
    print(ratio)