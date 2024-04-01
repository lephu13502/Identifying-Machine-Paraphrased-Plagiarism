# Import libraries
import gradio as gr


# Prepare model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "jpelhaw/longformer-base-plagiarism-detection"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Deploy
title       = "Machine-Paraphrased Detection"
description = """
                HCMUS | Applications of Big Data - PhD. Nguyễn Ngọc Thảo, PhD. Bùi Duy Đăng\n
                Group Information:\n
                19127496 - Trương Quang Minh Nhật\n
                20127275 - Lê Nguyễn Nhật Phú\n
                20127344 - Võ Hiền Hải Thuận
              """
inputs      = gr.Textbox(label="Input")
outputs     = gr.Textbox(label="Output")

examples = [
    [
        "The North Pacific right whale seems to happen in two populaces. The populace in the eastern North Pacific/Bering Sea is amazingly low, numbering around 30 people. A bigger western populace of 100 to 200 seems, by all accounts, to be making due in the Sea of Okhotsk, yet next to no is thought about this populace. In this manner, the two northern right whale species are the most jeopardized of every extensive whale and two of the most imperiled creature species on the planet. In light of current populace thickness patterns, the two species are anticipated to end up wiped out inside 200 years. The Pacific species was truly found in summer from the Sea of Okhotsk in the west to the Gulf of Alaska in the east, for the most part north of 50°N. Today, sightings are exceptionally uncommon and for the most part happen in the mouth of the Sea of Okhotsk and in the eastern Bering Sea. In spite of the fact that this species is all around prone to be transitory like the other two species, its development designs are not known."
    ],
    [
        """On 12 June 1917, a long time before "Mutsu" was set down, Hiraga proposed a modified structure for the ship that mirrored the exercises from the Battle of Jutland that had happened the earlier year, and consolidated advances in kettle innovation. Given undertaking number A-125, his structure included an additional twin primary firearm turret, utilizing space and weight made accessible by the decrease of the quantity of boilers from 21 to 12, while the power continued as before. He diminished the auxiliary weapon from 20 firearms to 16, in spite of the fact that they were brought up in stature to improve their capacity to discharge amid overwhelming climate and to improve their bends of shoot. To expand the ship's insurance he proposed calculating the belt protection outwards to improve its protection from flat flame, and expanding the thickness of the lower deck defensive layer and the torpedo bulkhead. Hiraga additionally intended to add against torpedo lumps to improve submerged security. He assessed that his ship would dislodge as much as "Nagato", in spite of the fact that it would cost around a million yen more. Hiraga's progressions would have impressively deferred "Mutsu"s fruition and were dismissed by the Navy Ministry."""
    ],
    [
        "On 24 February 2022, Russia invaded Ukraine in a major escalation of the Russo-Ukrainian War, which began in 2014. The invasion has likely resulted in tens of thousands of deaths on both sides and caused Europe's largest refugee crisis since World War II, with an estimated 8 million people being displaced within the country by late May as well as 7.8 million Ukrainians fleeing the country as of 8 November 2022. Within five weeks of the invasion, Russia experienced its greatest emigration since the 1917 October Revolution. The invasion has also caused global food shortages."
    ],
    [
        "Plagiarism is the representation of another's language, thoughts, ideas, or expressions as one's own original work. In educational contexts, there are differing definitions of plagiarism depending on the institution. Plagiarism is considered a violation of academic integrity such as truth and knowledge through intellectual and personal honesty in learning, teaching, research, fairness, respect and responsibility, and a breach of journalistic ethics. It is subject to sanctions such as penalties, suspension, expulsion from school or work, substantial fines and even imprisonment."
    ],
]


def inference(input):
    input_ids = tokenizer.encode(input, add_special_tokens=True, return_tensors="pt")
    example = {'input_ids': input_ids}
    answer = model(**example).logits.softmax(dim=-1).tolist()[0]
    predict = "Original" if answer[0] > answer[1] else "Plagiarism"
    return predict


gr.Interface(inference, inputs, outputs, title=title, description=description, examples=examples).launch(share=True)