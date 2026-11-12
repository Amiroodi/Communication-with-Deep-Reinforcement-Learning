# RIAL on the Switch Riddle

A PyTorch reimplementation of **RIAL** (Reinforced Inter-Agent Learning) from
[*Learning to Communicate with Deep Multi-Agent Reinforcement Learning*](https://arxiv.org/abs/1605.06676)
(Foerster, Assael, de Freitas, Whiteson, NIPS 2016), applied to the Switch
(prisoners and light bulb) riddle.

n prisoners are placed in the interrogation room one at a time, at random. The only way
they can communicate is a one-bit light switch. They must learn, from reward alone,
a protocol that lets one of them announce when everyone has visited the room.

**What's included**
- Vectorised Switch environment
- RIAL with parameter sharing: a recurrent Q-network (2-layer GRU) with separate
  Q-heads for actions and messages, trained with DQN and no experience replay
- Training curve of normalised reward against epochs
- Tools to inspect behaviour: day-by-day episode traces and a table of the learned protocol

**Run it**
- Locally: `pip install -r requirements.txt` then `python rial_switch.py --n 3 --epochs 5000`
- In Colab: open `rial_switch.ipynb`

This is an independent, unofficial implementation. It covers RIAL on the Switch task only,
not DIAL or the MNIST games. Some details the paper doesn't specify (e.g. the Switch input
network, gradient clipping) are my own choices. The original authors' code is at
https://github.com/iassael/learning-to-communicate.
