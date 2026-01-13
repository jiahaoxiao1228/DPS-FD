import csv
import json
import os
import random
import re
import time

import httpx
import spacy
from openai import OpenAI
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

API_SECRET_KEY = "Your API_SECRET_KEY"
BASE_URL = "Your BASE_URL"

SYSTEM_PROMPT = (
    "You are an expert language model specializing in generating diverse, high-quality product review texts for federated learning. "
    "You will carefully analyze a set of example reviews and identify corresponding domains and sentiment tendencies. "
    "While the examples may be biased toward certain domains, your task is to summarize them and infer additional potential domains. "
    "Your generation should reflect the authentic style and tone of product reviews: natural, varied, and customer-oriented. "
    "The output must ensure both domain diversity and linguistic diversity, avoiding repetitive templates while maintaining realism."
)


def generate_batch_public_data(batch_idx, batch_size, batch_words):
    user_prompt = (
        "Here are some representative review examples from clients:\n"
        "[Sample 1] The Slim Shady lp is pretty dope. The lyrics are awesome and the beats are catchy. His later releases rap about fame too much and the sound isn't as fresh. I listen to mostly metal but I still find myself putting this in alot. The best tracks are Role model, 97 Bonnie and Clyde, Brain damage and Guilty Conscience."
        "[Sample 2] If there is one book that you read in high school that you should read again later in life, this is it. In fact, I'm inclined to think that it shouldn't be taught until at least college because there is simply no way that any child in high school can truely appreciate this amazing novel. Whether you hated it or loved it when you read it in high school, or even if you didn't read it in high school, it is an outstanding story and Bradbury's writing is as smooth and clean as always.\n"
        "[Sample 3] This book was a wonderful work of literary art. I had to read this book for a class i was taking. I had already seen the movie, and based on that i wasn't excited to read the book. The movie was not very appealing. I am quite sure i wouldn't have read thiis book, based on the movie, if i wasn't forced to in my class.But am i happy i read it. The book was much more enticing and also was a lot more intellectual. It was a great peice of literary work and the morals were also important for everyday life.\n"
        "[Sample 4] Walt Whitman's literary reputation has been hyped by 'gay' critics (I suspect). I'm not totally hostile to free verse; in fact, I like T.S. Eliot, the most famous proponent of this genre, but Whitman wrote ramblingly, seemingly devoid of poetic coherence that the reader's creative imagination is left unstirred. I agree with Robert Frost, the truly great American poet, that writing poetry without the classic, conventional rules is like playing tennis--with no net.\n"
        "[Sample 5] This is a review of the free Kindle version.A Rogue's Life comes in at just under 1,800 locations.It's interesting reading Wilkie Collins's views on the work 20 years after writing it. His melancholy thoughts of his dead friends definitely grabs you.The rogue is an interesting sort, making his living with as little work as is possible. It's not too shocking that he ends up in some criminal activities. This definitely falls in the classic sensation category.The ending is pretty remarkable - I definitely chuckled about how and who the rogue ends up working for.The story is marred by some casual anti-Semitism, not that unusual for the time.\n"
        "[Sample 6] In the Mood for Love is a beautiful movie about love. And waht's perfect about it is that love in here is not sad or happy or stupid like in other movies, but above all love in here is human. Visually it is excellent (Christopher Doyle's name always means perfection in cinematography), the music is beautiful and the story is wonderful.It is a movie about two people that were in the mood to love each other but didn't.\n"
        "[Sample 7] This is a story about regular people getting over hurdles in their lives. Parenting, coping with limited financial resources, racism, being overweight are some of the everyday issues involved. However what makes this movie such a success is the sterling performance by the actors especially Halle Berry and Billy Bob Thornton. They definately brought the scenes to life and made us feel as though we were just peeking through our window and observing persons our neighborhood. They made the audience feel every emotion a particular scene conveyed. Definately Oscar material.\n"
        "[Sample 8] I saw this at the 2000 SF International Film Festival. It's an interesting film-a Chinese Vertigo, basically. The one thing I really disliked was that several of the main plot devices were based on situations that were EXTREMELY hard to swallow . I loved the imagery, though. It has a really gritty, modern edge to it that is pretty unusual for Chinese film. I definitly recommend this one, I'm buying it myself. The bad (ie:improbable) parts are bad enough (ie:improbable enough) to be enjoyed.\n"
        "[Sample 9] I was so disappointed by this purchase and this film. Being a fan of the book I was very excited to see how they translated it on to film. What I found out is that they didn't translate it at all. They LOOSELY followed the story line and took the liberty to add and change which in my opinion, changed the story all together. I threw this movie away and would not reccomend it to any true Story of O fan.\n"
        "[Sample 10] ....never mind.I ain't finishing that sentence!I just wanna say that the CD made me wanna get up and dance....no!Wait!It DID make me get up and dance!It's a necessary addition to my all too small CD collection.\n"
        "[Sample 11] It's hard to live up to your past, but when you have created an album as engaging as 'Lost Souls' expectations will forever be raised.'Some Cities' and even 'The Last Broadcast' can't match the depth of Doves debut album. Tracks like 'Black and White Town' and the title track are deginitely worth listening to, but I can't see this album dominating my cd player (ipod) the way that 'Lost Souls' did...does?\n"
        "[Sample 12] This is my second copy of this awsome CD. Its great listening and a nice mix of easy, relaxing music. I literally wore out my first copy. Often listen to it while driving and thoroughly enjoy the fabulous songs on it. I would highly recommend it.\n"
        "[Sample 13] I've always wanted to see BTS live, but they never play in my area. I knew they must be awesome live, and now I know. The whole cd is awesome. This is band which is limited when it comes to recording music, as they are much better live, which is one of the defining qualities of a great band. It seems like it'll be perfect from now on, and please keep their greatness like a secret, as it is more enjoyable that way.\n"
        "[Sample 14] The bag is well made for the price. Although the fit is loose for my Vanguard MG3 tripod from Costco, the bag looks good when I wear it over the shoulder.The cross section is 4.5' X 5', and it is smallest I could find in the net. And the Cameta camera shipped the bag in timely manner.\n"
        "[Sample 15] There are times when I need a bigger soldering iron to do some heavier soldering, and this big boy does a great job. It gets hot enough to handle the toughest jobs without any problems.\n\n"

        "Your task:\n"
        f"1. Identify the product domains and sentiment tendencies represented in the examples above, and also infer additional potential product domains that could reasonably exist, even if they are not explicitly shown in the examples.\n"
        f"2. Based on the examples' and inferred domains, generate exactly {batch_size} **realistic and diverse** unlabeled product review texts.\n"
        "3. Each review should be fluent, well-structured, reflecting the natural style of customer reviews. "
        "4. Each review should be about 90 words (minimum 40, maximum 180). Do not shorten because you need to output many; treat each independently.\n"
        "5. The reviews should **not** be templates or mechanically repeated. Maintain variability in tone, sentence structure, and vocabulary.\n\n"

        "**Output format:**\n"
        f"Output exactly {batch_size} reviews, each on a separate line. I will split the text with a newline character for saving. "
        "Ensure the quality of each generated sentence and do not sacrifice quality for the sake of quantity.\n"
        "Do not add any prefix, numbering, headers, or label — only the raw reviews.\n\n"
        # f"This is batch {batch_idx} of {total_target // batch_size}.\n"
        
        f"**Additional guidance for this batch:**\n"
        "Try to naturally include as many of the following words as possible in the reviews. "
        "These words are provided in their base forms (lemmas) and lowercase—feel free to adjust them "
        "as needed (plural, past tense, comparative, etc.) so that the reviews reflect authentic Amazon product feedback. \n"
        f"{', '.join(batch_words) if batch_words else 'No specific words for this batch.'}\n"

        "Begin now:"
    )

    client = OpenAI(base_url=BASE_URL, api_key=API_SECRET_KEY)
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.9,
            max_tokens=4096,
            presence_penalty=0.2,
            frequency_penalty=0.1
        )
        output_text = resp.choices[0].message.content.split("\n")
        data_list = [line.strip() for line in output_text if line.strip()]
        return data_list, len(data_list)

    except Exception as e:
        print(f"Error in batch {batch_idx + 1}: {e}")
        time.sleep(10)
        return 0


def save_generated_reviews(new_data, filename):
    if isinstance(new_data, str):
        new_data = [new_data]
    elif not isinstance(new_data, list):
        raise ValueError("Input must be a string or list of strings.")

    file_exists = os.path.isfile(filename)

    written = 0
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["text"])
        for line in new_data:
            clean_line = line.strip()
            if clean_line:
                writer.writerow([clean_line])
                written += 1
    return written


def get_existing_count(filename):
    if not os.path.isfile(filename):
        return 0
    with open(filename, encoding='utf-8') as texts:
        return sum(1 for _ in texts) - 1


def is_valid_text(text):
    text = text.strip()
    if re.match(r"^\d+\.\s", text):
        return -1

    if re.match(r"^[A-Za-z]", text):
        return True
    if re.match(r"^\d", text) and not re.match(r"^\d+\.\s", text):
        return True

    return False


def get_batch_words():
    global unused_words
    if len(unused_words) >= WORDS_PER_BATCH:
        batch_words = [unused_words.pop() for _ in range(WORDS_PER_BATCH)]
    else:
        batch_words = unused_words.copy()
        unused_words.clear()
        batch_words += random.sample(vocab_list, WORDS_PER_BATCH - len(batch_words))
    return batch_words


def update_unused_words(batch_words, generated_texts):

    global unused_words, used_words, appeared_words
    batch_words_set = set([w.lower() for w in batch_words])

    used_words |= batch_words_set

    unused_words = [w for w in unused_words if w.lower() not in batch_words_set]

    for text in generated_texts:
        doc = nlp(text)
        words_in_text = {token.lemma_.lower() for token in doc if token.is_alpha}
        appeared_words.update(batch_words_set & words_in_text)

    print(f"Total appeared words in generated texts: {len(appeared_words)}, Total used_words: {len(used_words)}, Total unused_words: {len(unused_words)}")


if __name__ == "__main__":
    random.seed(42)
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    # Set parameters
    VOCAB_FILE = "amazon_style_vocab.json"
    WORDS_PER_BATCH = 200
    batch_size = 20
    total_target = 59188
    pub_data_filename = "./gpt_all_domain_public_data_vocab.csv"

    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        vocab_list = json.load(f)

    random.shuffle(vocab_list)
    appeared_words = set()
    unused_words = vocab_list.copy()
    used_words = set()

    existing_count = get_existing_count(pub_data_filename)
    remaining = total_target - existing_count
    print(f"Existing data: {existing_count} reviews.")
    print(f"Need to generate: {remaining} more reviews.")

    for batch_idx in range(remaining // batch_size):
        print(f"\n[Batch {batch_idx + 1} / {remaining // batch_size}] Generating {batch_size} reviews...")

        # 1. Get the word set of this batch
        batch_words = get_batch_words()

        try:
            # 2.Call the generation function (add word constraint parameters)
            texts, num_text = generate_batch_public_data(
                batch_idx,
                batch_size,
                batch_words=batch_words
            )

            # 3. Verify and save
            valid_texts = []
            for text in texts:
                flag = is_valid_text(text)
                if flag == -1:
                    valid_texts = []
                    break
                elif flag:
                    valid_texts.append(text)
                else:
                    print(f"[Skipped] Invalid text (not starting with letter): {text[:50]}...")

            if valid_texts:
                written = save_generated_reviews(valid_texts, pub_data_filename)
                # 4. Update the list of uncovered words
                update_unused_words(batch_words, valid_texts)
            else:
                print("[Warning] All generated texts in this batch were invalid. No text saved.")
            print(f'number of output_text: {num_text}')
            df = pd.read_csv(pub_data_filename)
            print(f"The total number of saved texts: {len(df)}")
        except Exception as e:
            print(f"Error in batch {batch_idx + 1}: {e}")
            print("Waiting 10 seconds before retrying...")
            time.sleep(10)
