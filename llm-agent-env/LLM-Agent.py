#Loading a Pre-Trained Model
from transformers import AutoModelForCausalLM, AutoTokenizer

#Load the tokenizer and model 
tokenizer = AutoTokenizer.from_pretrained("gpt2") #we use the GPT-2 model
model = AutoModelForCausalLM.from_pretrained("gpt2")

#Sample input text
input_text = "Hi world !"
#my name is Ishfaaq and this is my text. Usually it should be very long but I don't have time to find long text to copy or integrate so this is a template. In the future, it will be beter to introduce a file text and importing it here."

#TOkenize input
input_ids = tokenizer.encode(input_text, return_tensors='pt') #encode input_text

#Generate response
outputs = model.generate(input_ids = input_ids, max_length = 150, temperature=0.7, repetition_penalty=1.2) #generate sequence
response = tokenizer.decode(outputs[0], skip_special_tokens=True) 

print(" the response is : ", response)

#there are still some improvement to do : Setting the attention  mask, pad token id ...


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict',methods = ['POST'])
def predict():
    input_text = request.json['text']
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids = input_ids, max_length = 150, temperature=0.7, repetition_penalty=1.2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify(response=response)

if __name__ == '__main__':
    app.run(debug=True)
