# Machine-Paraphrased Detection
Machine-Paraphrased Detection with [Longformer model](https://huggingface.co/jpwahle/longformer-base-plagiarism-detection) and deployed using [Gradio](https://github.com/gradio-app/gradio).

## Information

Ho Chi Minh City University of Science

Applications of Big Data - PhD. Nguyễn Ngọc Thảo, PhD. Bùi Duy Đăng

No. | Student ID | Student Name
--- | :---:      | ---
1   | 19127496   | Trương Quang Minh Nhật
2   | 20127275   | Lê Nguyễn Nhật Phú
3   | 20127344   | Võ Hiền Hải Thuận

## How to run the deploy
1. Clone repository.
<pre>git clone repo-link</pre>
2. Install Python.
3. Install necessary libraries.
<pre>pip install -r requirements.txt</pre>
4. Run the deploy, the first time downloading the model would take about 5 minutes, the next time would not need to reload.
<pre>python app.py</pre>
5. Browse the deploy on Localhost via the link http://localhost:7860, or the Public link generated in Command prompt.
6. Enjoy 🙂

## References

[[1](https://huggingface.co/jpwahle/longformer-base-plagiarism-detection)] Longformer-base for Machine-Paraphrase Detection.

[[2](https://arxiv.org/abs/2103.11909)] Jan Philip Wahle, Terry Ruas, Tomáš Foltýnek, Norman Meuschke, Bela Gipp: Identifying Machine-Paraphrased Plagiarism.

[[3](https://github.com/gradio-app/gradio)] Gradio: Build Machine Learning Web Apps — in Python.