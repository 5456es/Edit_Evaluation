from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 模型和tokenizer的保存路径
model_dir = "/home/bizon/zns_workspace/24_09_Evaluation/EasyEdit/Method_Test/edited_models/test_basic"

# 加载保存好的GPT-2模型
model = GPT2LMHeadModel.from_pretrained(model_dir)

# 加载GPT-2的tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

# 测试生成
input_text = "Paris, the capital of France or England? France. Ray Charles, piano or violin?" 
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs,max_length=50,num_return_sequences=1)

# 打印生成结果
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
