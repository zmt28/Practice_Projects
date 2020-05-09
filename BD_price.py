import bs4, requests

def getBookDepository(ProductUrl):
    res = requests.get(ProductUrl)
    res.raise_for_status()

    soup = bs4.BeautifulSoup(res.text, 'html.parser')
    elems = soup.select('body > div.page-slide > div.content-wrap > div > div > div.item-wrap > div.item-block > div.item-tools > div > div.price-info-wrap > div > div.price.item-price-wrap.hidden-xs.hidden-sm > span.sale-price')
    return elems[0].text



price = getBookDepository("https://www.bookdepository.com/Automate-Boring-Stuff-With-Python-Al-Sweigart/9781593275990?ref=pd_detail_1_sims_b_p2p_1")
print('The Price is: '+ price)
