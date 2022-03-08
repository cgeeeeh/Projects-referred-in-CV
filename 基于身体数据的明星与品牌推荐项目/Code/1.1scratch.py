from selenium import webdriver
import time
import csv
import sys


def Parse_ModelINFO1(INFO):
    INFO=INFO.split('\n')
    #print(INFO)
    result=[]
    for i in range(len(INFO)):
        if i==2:
            result.append(len(INFO[i].split(':')[1]))
        else:
            result.append(INFO[i].split(':')[1])
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
with open('MODEL.csv','a',newline='') as csvFile:
    writer = csv.writer(csvFile)
    #writer.writerow(["Page","Row","Line","NO.", "Category", "Level","City","Height","Chest","Waist","Hips","Suit","Shoes","Skin","Eyes","Hair","advertisement","popularity"])

    checki=0
    checkj=0
    page=1
    while page<234:
        if page<231:
            page=page_switch(page)
            continue
        for i in range(1,6):
            if page==19 and i<1:
                continue
            for j in range(1,5):
                if page==19 and i==1 and j<1:
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
                    element=browser.find_element_by_xpath('//*[@id="ModelInfoShow_6"]')
                    Text1=Parse_ModelINFO1(element.text)
                    element=browser.find_element_by_xpath('//*[@id="ModelInfoShow_4"]/table/tbody/tr[2]/td/table/tbody')
                    Text2=Parse_ModelINFO2(element.text)
                    #print(Text1,Text2)
                    Text=[page,i,j]+Text1+Text2
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




