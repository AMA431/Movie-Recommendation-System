# Movie Recommendation System

A simple movie recommendation system built using **Collaborative Filtering**.
It suggests movies to a user based on the ratings of similar users.

---

## Overview

This project uses **User-Based Collaborative Filtering** to recommend movies.
It compares users using **Cosine Similarity** and recommends movies that
similar users liked but the target user hasn't watched yet.

The goal is to demonstrate a core concept of AI/ML — **recommender systems** —
in the simplest possible way.

---

##  Features

-  Builds a user–movie rating matrix from raw ratings
-  Computes similarity between users using cosine similarity
-  Recommends top-N movies for any given user
-  Skips movies the user has already watched
-  Simple command-line output

---

##  Technologies Used

| Tool | Purpose |
|------|---------|
| **Python 3.8+** | Programming language |
| **pandas** | Data handling and pivot tables |
| **scikit-learn** | Cosine similarity computation |

---

## 📁 Project Structure
