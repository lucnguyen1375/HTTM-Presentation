# Assignment 06 — RNN (Hệ thống thông minh)

Sinh viên: Giáp Minh Hiếu — B23DCCN299 — D23CTPM01-B

## Nội dung

- `notebook/` — 5 notebook + mã nguồn `.py`
  - `00_rnn_concepts.ipynb` — kiến trúc RNN, W/U/V, BPTT (numpy), đối chiếu `nn.RNN` ↔ `SimpleRNN`
  - `01-metals-rnn-pytorch.ipynb` / `02-metals-rnn-keras.ipynb` — dự báo giá vàng/bạc (PyTorch + Keras)
  - `03-amzn-rnn-pytorch.ipynb` / `04-amzn-rnn-keras.ipynb` — dự báo giá AMZN (PyTorch + Keras)
  - `*-nb-source.py` — mã nguồn dựng notebook
- `test-e2e.py` — kiểm thử end-to-end
- `slides/rnn.pptx` — slide chuyên đề RNN

## Kỹ thuật chính

Dự báo **Δ residual** (giá = giá hôm nay + Δ̂, guard ±3σ) thay vì dự báo mức giá trực tiếp.
