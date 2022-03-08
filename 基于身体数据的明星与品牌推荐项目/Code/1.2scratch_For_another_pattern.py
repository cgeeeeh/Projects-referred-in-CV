from selenium import webdriver
import time
import csv
import sys


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
    #print(result)
    return result

def Parse_ModelINFO2(INFO):
    INFO=INFO.split('\n')
    if INFO[1]=='影视作品:':
        return ['',INFO[-1]]
    else:
        return [INFO[1], INFO[-1]]

def page_switch(page):
    try:
        element = browser.find_element_by_xpath('//*[@id="AspNetPager1"]/a[12]')
        browser.execute_script('arguments[0].click();', element)
        page = page + 1
        time.sleep(2)
    except:
        print("meet error during page switch", page)
        csvFile.close()
        exit()
    return page

browser = webdriver.Chrome()
browser.get("http://www.chinaeve.com/ModelList.aspx")
#print(browser.page_source)
with open('MODEL1.csv','a',newline='',encoding='UTF-8') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(["Page","Row","Line","NO.", "Category", "Level","City","Height","Chest","Waist","Hips","Suit","Shoes","Skin","Eyes","Hair","advertisement","popularity"])

    checki=0
    checkj=0
    page=1
    while page<234:
        if page<99:
            page=page_switch(page)
            continue
        for i in range(1,6):
            if page==1 and i<1:
                continue
            for j in range(1,5):
                if page==1 and i==1 and j<1:
                    continue

                try:
                    #点击模特元素
                    element=browser.find_element_by_xpath('//*[@id="DList"]/tbody/tr['
                                                          +str(i)
                                                          +']/td['
                                                          +str(j)
                                                          +']/div/ul/li[2]/span[1]/a')
                    browser.execute_script('arguments[0].click();',element)

                    #切换窗口
                    nowhandle=browser.current_window_handle
                    allhandles=browser.window_handles
                    for handle in allhandles:
                        if handle != nowhandle:
                            browser.switch_to.window(handle)
                            #print('now register window!')
                    #读取模特信息
                    element=browser.find_element_by_xpath('//*[@id="FV"]/tbody/tr/td/table/tbody')
                    Text1=Parse_ModelINFO1(element.text)
                    #element=browser.find_element_by_xpath('//*[@id="ModelInfoShow_4"]/table/tbody/tr[2]/td/table/tbody')
                    #Text2=Parse_ModelINFO2(element.text)
                    #print(Text1,Text2)
                    Text=[page,i,j]+Text1
                    writer.writerow(Text)
                    print(Text)
                    browser.close()
                    allhandles=browser.window_handles
                    browser.switch_to.window(allhandles[0])

                except Exception as e:
                    print("meet an error during scratch process")
                    browser.close()
                    allhandles = browser.window_handles
                    browser.switch_to.window(allhandles[0])
                    pass

        page=page_switch(page)




