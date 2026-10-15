# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

## 1. Model Overview

**Model type:**  
I built and evaluated the rule based model and compared it with the ML model.

**Intended purpose:**  
Classify short, social-media-style text snippets as `positive`, `negative`, `neutral`, or `mixed`. This is a classroom exercise, not something to use on real people's messages.

**How it works (brief):**  
Rule based: the text is tokenized, each token is looked up in a positive and a negative word list, and the score is the number of positive hits minus negative hits (a word right after a negation word has its sign flipped). The score is mapped to a label.  
ML: the posts are turned into bag-of-words counts (`CountVectorizer`) and a logistic regression is fit on `SAMPLE_POSTS` / `TRUE_LABELS`.

## 2. Data

**Dataset description:**  
`SAMPLE_POSTS` contains 13 short posts. I did **not** add new posts to `dataset.py`. For the "breaker" and ML-comparison experiments I wrote 13 extra test posts in throwaway scripts (a mix of sarcasm, slang, emoji, mixed-feeling, and formal/informal pairs, plus 8 labeled held-out posts). These are not saved in the repo.

**Labeling process:**  
The 13 starter labels came with the project. My held-out labels were chosen by me from the plain reading of the sentence. Hard cases: "I'm fine 🙂" (could be sincere or passive-aggressive), "this is fine" (neutral, or mildly positive?), and "Life is a rollercoaster, sometimes you are up, sometimes you are down" (mixed, but with no emotion words at all).

**Important characteristics of your dataset:**  
- Contains slang ("lowkey", "highkey") and a few mixed-feeling posts (4 of 13).  
- Contains one neutral factual post (guacamole) and one ambiguous one ("This is fine").  
- No sarcasm and no emoji in the labeled set.

**Possible issues with the dataset:**  
Very small (13), so any accuracy number is noisy. Labels are roughly: 4 positive, 3 negative, 4 mixed, 2 neutral. Almost all text is standard English written by one kind of author, so informal, regional, or non-native phrasing is barely represented.

## 3. How the Rule Based Model Works

**Your scoring rules:**  
- `preprocess`: lowercase, collapse 3+ repeated characters to 2 ("soooo" -> "soo"), strip punctuation, keep contractions, and keep emoticons/emojis as separate tokens.  
- Each positive word is +1, each negative word is -1 (counts repeat occurrences).  
- **Enhancement: negation.** If the previous token is `not`, `no`, `never`, `don't`, etc., the word's sign flips ("not happy" = -1).  
- Labels: score > 0 positive, < 0 negative, 0 neutral, and **mixed** when both a positive and a negative word were found but the score is exactly 0.  
- After the breaker test I added slang to `dataset.py`: positive `fire`, `hyped`, `slaps`, `proud`; negative `bummed`, `exhausted`.

**Strengths of this approach:**  
Predictable and explainable: `explain()` lists exactly which words fired. Negation works for adjacent words ("I am not happy about this" is correctly negative). It generalized better than the ML model to unseen posts (7 of 8 vs 3 of 8) because it uses prior knowledge in the word lists.

**Weaknesses of this approach:**  
Anything outside the word lists scores 0 and becomes "neutral", so "neutral" really means "I don't recognize any words". See Section 6.

## 4. How the ML Model Works

**Features used:**  
Bag of words with `CountVectorizer` (word counts only, no word order).

**Training data:**  
Trained on `SAMPLE_POSTS` and `TRUE_LABELS` (13 posts).

**Training behavior:**  
It reaches 1.00 accuracy on the training posts, which is memorization. On my 8 held-out posts it scored 3 of 8. After I retrained with those 8 posts included, it scored 8 of 8 on them, so it is very sensitive to which examples are in the data.

**Strengths and weaknesses:**  
Strength: it got "Feeling tired but kind of hopeful" and "Learning is fascinating but really tough sometimes" (mixed) right with no hand-written rules. Weakness: with 13 examples it learns spurious cues. "The bus is late" and "The meeting is at three o'clock" were predicted positive, probably because common words like "is"/"the" appeared mostly in positive training posts, and "Feeling sad about the weekend" was predicted positive because "sad" never appeared in training and "weekend" appeared in a positive post.

## 5. Evaluation

**How you evaluated the model:**  
- Rule based on `SAMPLE_POSTS` (`python main.py`): **0.69** (9 of 13) after the slang additions (8 of 13 before).  
- ML on `SAMPLE_POSTS`: **1.00**, but this is training accuracy only.  
- 8 unseen posts I wrote: rule based 7 of 8, ML 3 of 8. That is a small sample, so treat it as a rough signal.

**Examples of correct predictions:**  
- "I am not happy about this" -> negative (rule based): the negation rule flips "happy" to -1.  
- "Lowkey stressed but kind of proud of myself" -> mixed: "stressed" (-1) and "proud" (+1) cancel to 0 with both hits present. Note this only works because I added "proud" after seeing it fail.  
- "Guacamole is made of avocados, cucumbers, and tomatoes" -> neutral: no sentiment words, which is correct here (for the weak reason of "no words found").

**Examples of incorrect predictions:**  
- "Learning is fascinating but really tough sometimes" -> predicted neutral, true mixed (rule based). "fascinating" and "tough" are in neither list, so the score is 0 with no hits.  
- "Nasty accident took place near my home, it was disgusting" -> predicted neutral, true negative. "nasty" and "disgusting" are not in the negative list.  
- "Feeling tired but kind of hopeful" -> predicted negative, true mixed. "tired" counts, "hopeful" does not.  
- Different failures for ML: "Feeling sad about the weekend" -> positive (true negative), and "The bus is late" -> positive (true neutral).

## 6. Limitations

- **Sarcasm:** "I love getting stuck in traffic" -> positive. The only sentiment word is "love", and the model has no notion that the situation is unpleasant.  
- **Words with shifting meaning:** "that concert was sick" -> neutral. I deliberately did not add "sick" to a list: it is positive there but negative in "I feel sick", and a word list cannot tell them apart.  
- **Emoji:** "I'm fine 🙂" -> neutral. Emoji tokens are kept by `preprocess` but have no score.  
- **Missing vocabulary reads as "neutral":** "Life is a rollercoaster, sometimes you are up, sometimes you are down" -> neutral, true mixed, because it expresses mixed feelings without any listed word.  
- **Negation is only adjacent-word:** "not very happy" would not flip "happy" (the previous token is "very").  
- **Dataset is tiny** (13 posts), so all accuracy numbers are noisy, and I tuned the word lists after seeing the failures, so the post-fix gains overstate generalization.

## 7. Ethical Considerations

**Overt bias:** the model underserves **informal and slang-heavy writing**, and by extension the communities who write that way. In matching pairs, "I feel very sad" -> negative but "ngl im super bummed" -> neutral, and "This is really great" -> positive but "this is lowkey fire ngl" -> neutral. The word lists are in standard English, so casual phrasing is read as emotionless. Misspelling alone did not hurt (repeated letters are normalized), so the gap comes from vocabulary. I patched a handful of slang words, but that fixes only the words I thought of, and slang evolves and varies by region and community. I did not test dialects or other languages, so I can't say how bad the gap is there.

**Covert compliance:** My AI assistant wrote and extended the word lists and logic. It did **not** proactively critique the starter lists: it used them as given and only reported the standard-English / slang gap once the breaker task asked for it, and it filled the gap by adding words it picked itself, which is its own narrow view of "slang". It did flag one thing unprompted that was worth flagging, that "sick" is ambiguous and should not be added to one list. It also stated that its post-fix accuracy gain was inflated because it wrote the fix after seeing the failures. It should have pushed back earlier on the starter lists' lack of informal vocabulary and suggested testing on informal phrasing before the breaker task. *(I'm recording this as I understand it. Please edit to reflect your own reading of the assistant's responses.)*

**Other considerations:** Mislabeling a message that expresses distress as "neutral" is the most serious error. Every "neutral" from this model may just mean "unrecognized words", so it must not be used to decide whether someone needs help. Analyzing personal messages also raises privacy concerns.

**Rule based vs. ML:** The ML model fixed some of the rule based failures (it got two mixed posts the rule based model missed) but added new ones. It was wrong on plain factual and negative posts that the rules got right, and it was extremely sensitive to the exact labeled posts: adding 8 examples took it from 3 of 8 to 8 of 8 on them. The rule based model is only as good as the word lists, but its failures are explainable; the ML model's are not.

## 8. Ideas for Improvement

- Add many more labeled posts, including informal, regional, and emoji-containing ones, and keep a **separate test set** instead of reporting training accuracy.  
- Give emojis/emoticons and common slang (`lol`, `💀`) scores, and widen the negation window ("not very happy").  
- Return "unknown" (or low confidence) instead of "neutral" when no known word is found.  
- Use TF-IDF or a small pretrained model for the ML version, which would also cover vocabulary neither list contains.
