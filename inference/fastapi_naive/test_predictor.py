from predictor import Predictor

if __name__ == "__main__":
    from datasets import load_dataset

    dataset_id = "samsum"
    dataset = load_dataset(dataset_id)
    dataset["test"]["dialogue"][0]

    model_load_path = "weights/checkpoints-for-summarization/assets"
    task = "summarization"
    p = Predictor(model_load_path=model_load_path, task=task)
    text = "Hannah: Hey, do you have Betty's number?\nAmanda: Lemme check\nHannah: <file_gif>\nAmanda: Sorry, can't find it.\nAmanda: Ask Larry\nAmanda: He called her last time we were at the park together\nHannah: I don't know him well\nHannah: <file_gif>\nAmanda: Don't be shy, he's very nice\nHannah: If you say so..\nHannah: I'd rather you texted him\nAmanda: Just text him 🙂\nHannah: Urgh.. Alright\nHannah: Bye\nAmanda: Bye bye"
    prediction = p.predict(prompt=text)
    print(f"prediction = {prediction}")

    from datasets import load_dataset

    dataset_id = "rungalileo/20_Newsgroups_Fixed"
    dataset = load_dataset(dataset_id)
    dataset["test"]["text"][0]

    model_load_path = "weights/checkpoints-for-classification/assets"
    task = "classification"
    p = Predictor(model_load_path=model_load_path, task=task)
    text = "I am a little confused on all of the models of the 88-89 bonnevilles.\nI have heard of the LE SE LSE SSE SSEI. Could someone tell me the\ndifferences are far as features or performance. I am also curious to\nknow what the book value is for prefereably the 89 model. And how much\nless than book value can you usually get them for. In other words how\nmuch are they in demand this time of year. I have heard that the mid-spring\nearly summer is the best time to buy."
    prediction = p.predict(prompt=text)
    print(f"prediction = {prediction}")