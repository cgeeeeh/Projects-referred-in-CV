



def Parse_ModelINFO1(INFO):
    INFO=INFO.split('\n')
    #print(INFO)
    result=[]
    for i in range(len(INFO)):
        if i in (1,3,5,12,14):
            continue
        elif i==0:
            result.append(INFO[i].split('：')[1].strip())
        elif i==4:
            if '-' in INFO[i]:
                result.append(INFO[i].split('：')[1].split('-')[1])
            else:
                result.append(INFO[i].split('：')[1])
        elif i==7:
            if '-' in INFO[i]:
                tmp=INFO[i].split('：')[1].split('-')
                for item in tmp:
                    result.append(item)
            else:
                result.append('')
                result.append('')
                result.append('')
        else:
            result.append(INFO[i].split('：')[1])
    result.insert(2,'')
    result.append('')
    result.append('')
    print(result)
    return result


INFO='编号：J1017968	\n' \
     '姓名：Anna.L\n' \
     '类别：外籍模特\n' \
     '国籍：\n' \
     '城市：广东-广州\n' \
     '性别：女\n' \
     '身高：171\n' \
     '三围：82-59-88\n' \
     '肩宽：\n' \
     '鞋码：39\n' \
     '肤色：黄色\n' \
     '眼睛色：蓝色\n' \
     '体重：\n'\
     '头发：長发\n' \
     '学历：高中以下'

Parse_ModelINFO1(INFO)