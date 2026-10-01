from transformers import BertTokenizer, BertForQuestionAnswering
import torch

import warnings
warnings.filterwarnings("ignore")

# squad veri seti uzerinde ince ayar yapilmis bert dili modeli
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# bert tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# soru cevaplama gorevi icin ince ayar yapilmis bert modeli
model = BertForQuestionAnswering.from_pretrained(model_name)

# cevaplari tahmin eden fonksiyon
def predict_answer(context, question):
    """
    context = metin
    question = soru
    Amac: metin icerisinden soruyu bulmak
    
    1) tokenize
    2) metnin icerisinde soruyu ara
    3) metnin icerisinde sorunun cevabinin nerelerde olabileceginin skorlarini return et
    4) skorlardan tokenlarin indeksleri hesapladik
    5) tokenlari bulduk yani cevabi bulduk
    6) okunabilir olmasi icin tokenlardan string'e cevirdik
    """
    
    # metni ve soruyu tokenlara ayiralim ve modele uygun hale getirelim
    encoding = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
    
    # giris tensorlerini hazirla
    input_ids = encoding["input_ids"]  # tokenlerin id
    attention_mask = encoding["attention_mask"]  # hangi tokenlarin dikkate alinacagini belirtir
    
    # modeli calistir ve skorlari hesapla
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, return_dict=False)
        start_scores, end_scores = outputs[0], outputs[1]
        
    # en yuksek olasiliga sahip start ve end indekslerini hesapliyor
    start_index = torch.argmax(start_scores, dim=1).item()  # baslangic indeks
    end_index = torch.argmax(end_scores, dim=1).item()      # bitis indeksimiz
    
    # token id lerini kullanarak cevap metnini elde edelim
    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start_index: end_index + 1])
    
    # tokenlari birlestir ve okunabilir hale getir
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    
    return answer

# ornek kullanim
question = "What is the capital of France?"
context = "France, officially the French Republic, is a country whose capital is Paris"

answer = predict_answer(context, question)
print(f"Question: {question}")
print(f"Answer: {answer}")