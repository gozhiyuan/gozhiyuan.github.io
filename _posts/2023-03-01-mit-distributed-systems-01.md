---
layout: post
title: Distributed Systems Introduction and MapReduce Lecture 1
subtitle: Distributed Systems Lecture 1
categories: MIT-Distributed-Systems-2021
tags: [distributed-systems]
banner: "/assets/images/banners/yuanpang-wa-valley.jpg"
---

[Course Link](https://pdos.csail.mit.edu/6.824/schedule.html)

## 🌐 What is a Distributed System?

- A **distributed system** is a collection of computers that communicate over a network to perform a task together.

### Examples
- 📱 Popular app backends (e.g., for messaging)
- 🌐 Large websites
- 🖧 Domain Name System (DNS)
- 📞 Phone systems

- These systems often use **services that are themselves distributed**:
  - 🗄️ Databases
  - 🔄 Transaction systems
  - 📊 Big data processing frameworks
  - 🔐 Authentication services

- Sometimes, the **distributed nature is the service itself**, like:
  - ☁️ Cloud systems such as Amazon Web Services (AWS)

This class focuses on infrastructure services, like:
  - 💾 Storage for large websites
  - 🔄 MapReduce
  - ↔️ Peer-to-peer sharing

---

## 🛠️ Why Build Distributed Systems?

- **Performance**: Allows for parallel processing, increasing capacity and performance.
- **Fault Tolerance**: Replicating data and services ensures continued operation even with some failures.
- **Physical Distribution**: Useful for geographically dispersed devices like sensors or bank branches.
- **Security**: Isolating components across different computers enhances security.

---

## ⚙️ Challenges of Distributed Systems

- **Concurrency**: Multiple computers operating together introduce complex interactions, making systems hard to design and debug.
- **Partial Failures**: Distributed systems can have partial failures, where some components continue to work while others fail.
- **Performance Bottlenecks**: Bottlenecks in network, storage, or other components can slow down the system.

---

## 🎓 Why Study Distributed Systems?

- **Intellectual Challenge**: Offers engaging, complex problems that require innovative solutions.
- **Widespread Use**: Essential to modern computing, powering large-scale websites and cloud services.
- **Research Opportunities**: Many open problems provide opportunities for research and development.
- **Practical Experience**: Building these systems provides valuable, hands-on experience with modern technology.

---

## 🏫 Course Structure

- **Lectures**: Cover key ideas, discuss research papers, and provide lab guidance. Lectures are recorded and available online.
- **Papers**: Almost every lecture has an assigned paper. Students submit responses to questions on the paper and ask their own questions.
- **Exams**: A midterm and final exam focus on material from the papers and labs.
- **Labs**: Hands-on experience with techniques in distributed systems programming.

    - Lab Topics:
      - **Lab 1**: A distributed big data framework (e.g., MapReduce)
      - **Lab 2**: Client/server communication over unreliable networks
      - **Lab 3**: Fault tolerance with replication (Raft)
      - **Lab 4**: Fault-tolerant databases
      - **Lab 5**: Scalable database performance via sharding

- **Final Project**: Optional group project, substitutes for Lab 5. Teams design, build, and evaluate a distributed system.

---

## 🔍 Main Topics

- **Storage**
- **Communication**
- **Computation**
- **Fault Tolerance**
- **Consistency**
- **Performance**
- **Tradeoffs**
- **Implementation**

---

## ⚡ Fault Tolerance

- **Inevitability of Failures**: Failures are a given in large-scale distributed systems.
- **Hiding Failures**: Systems should mask failures from users for reliability.
- **Availability and Recoverability**:
    - High-availability systems keep running despite failures.
    - Recoverable systems can restart without data loss or inconsistency.

- **Tools for Fault Tolerance**:
    - Non-volatile storage
    - Replication

---

## 🔄 Consistency

- **Defining Behavior**: Requires well-defined read and write behavior.
- **Challenges of Replication**: Maintaining identical replicas is difficult.
- **Strong vs. Weak Consistency**:
    - **Strong Consistency**: Ensures recent data, but can be expensive.
    - **Weak Consistency**: Sacrifices some guarantees for performance.

---

## 🚀 Performance

- **Scalable Throughput**: Adding servers should ideally increase throughput.
- **Challenges to Scalability**:
    - Load imbalance
    - Latency bottlenecks
    - Non-scalable operations

---

## ⚖️ Tradeoffs

- **The Triad of Fault Tolerance, Consistency, and Performance**: Balancing these three is challenging.
- **Communication Costs**: Consistency and fault tolerance often increase communication overhead.
- **Sacrificing Consistency for Speed**: Some systems trade consistency for performance.

---

## 🧩 Implementation Techniques

- **Remote Procedure Call (RPC)**: Abstracts communication between servers.
- **Threads**: Manages concurrency efficiently.
- **Concurrency Control**: Prevents data corruption with mechanisms like locks.

---

## 📖 Case Study: MapReduce

- **Context**: Designed for large-scale computations at Google, such as building search indexes.
- **Goals**: 
    - Handle multi-hour computations on large datasets.
    - Simplify programming for non-experts in distributed systems.
- **Programmer's View**: Programmers write simple Map and Reduce functions; the framework handles distribution.

---

### Example: Word Count

1. **Map Function**: Splits a file into words, emits each word as a key with a value of 1.
2. **Reduce Function**: Sums occurrences of each word and emits the word with its count.

- **Scalability**: Both Map and Reduce tasks run in parallel for speedup.
- **Hidden Details**: Manages code distribution, task tracking, and data shuffling.

---

## 🛠️ Implementation Details

- **Google File System (GFS)**: Stores input/output data across multiple servers.
- **Coordinator**: Assigns and monitors tasks, handles failures.
- **Workers**: Execute Map and Reduce tasks.

---

## 🏁 Conclusion

- **Strengths**:
    - Made cluster computation accessible.
    - Scalable and simple programming model.

- **Limitations**:
    - Not the most efficient or flexible.
    - Limited to a specific data flow pattern, lacks real-time support.

---

## 🌐 What is MapReduce? Paper Deep Dive

MapReduce is a programming model and implementation developed by Google for processing very large datasets on clusters of computers. It's well-suited for tasks that can take hours on terabytes of data, like indexing the web, sorting data, and analyzing web structure. MapReduce jobs efficiently leverage thousands of computers, which was essential for rapid results at Google.

### 🎯 MapReduce Goals

* **Ease of Use for Non-Specialists**: MapReduce enables programmers without distributed systems expertise to write programs for large clusters without managing complexities like parallelization, data distribution, fault tolerance, or load balancing.
* **Scalable Throughput**: Ideally, adding more computers to a MapReduce job proportionally increases throughput, providing scalable speed-up.

## 🛠️ How MapReduce Works

A MapReduce program consists of two user-defined functions: **Map** and **Reduce**. The MapReduce framework manages the rest of the distributed computation.

1. **Input Splitting**: The data, stored on a distributed file system (e.g., GFS), is split into smaller chunks called **splits** (typically 16MB-64MB).
2. **Map Tasks**: The **MapReduce master** assigns map tasks to worker machines. Each map task processes one input split, generating intermediate key/value pairs, which are stored on the worker’s local disk.
3. **Shuffle**: After all map tasks complete, the master assigns **reduce tasks**. The shuffle phase gathers intermediate values with the same key from all map tasks and sends them to the appropriate reduce worker.
4. **Reduce Tasks**: Each reduce worker sorts the key/value pairs by key. The user-defined Reduce function merges all values for each unique key, producing the final output stored in GFS.

## 📈 MapReduce in Practice

### 🔧 Optimizations

* **Data Locality**: To reduce network traffic, the master assigns map tasks to machines with local replicas of the input split, minimizing data transfer.
* **Task Granularity**: Splitting data into many map and reduce tasks (often more than available workers) ensures dynamic load balancing as faster workers can handle additional tasks.
* **Backup Tasks**: To mitigate "straggler" workers, backup tasks are scheduled for long-running tasks, reducing overall job completion time.

### ⚙️ Fault Tolerance

MapReduce is resilient to machine failures, common in large clusters.

* **Worker Failure**: If a worker fails, the master reassigns its tasks. Map tasks are re-executed (as the intermediate data is lost), while completed reduce tasks are unaffected (output stored on GFS).
* **Master Failure**: Currently, a master failure aborts the job; clients may retry the job. Periodic checkpoints could allow recovery without restarting.

To ensure correct output with failures, MapReduce relies on:

* **Atomic Commits**: Temporary files are used for map and reduce outputs until tasks complete. The master tracks these files, ensuring only one successful reduce execution’s output is in the final file.
* **Deterministic Functions**: Map and reduce functions must be deterministic, producing consistent output regardless of the worker. This means no side effects, external interactions, or randomness.

### 🚀 Refinements

* **Combiner Function**: For associative Reduce functions (like word count), a Combiner function can partially merge intermediate data before network transfer, saving bandwidth.
* **Custom Input/Output Types**: MapReduce supports custom input formats (text files, key/value pairs) and output types.
* **Handling Bad Records**: MapReduce can skip records causing deterministic crashes, enabling the job to continue despite specific errors.

### 💼 Google's Use of MapReduce

Google used MapReduce for:

* Large-scale machine learning and clustering
* Data analysis (Google News, Froogle)
* Extracting popular query data
* Large-scale graph computations

Google even rewrote its indexing system using MapReduce, simplifying operations and making it more efficient.

## ⚠️ Limitations

* **Restricted Data Flow Pattern**: MapReduce’s single Map/Reduce pattern limits it to simple data flows. Complex pipelines need multiple MapReduce jobs.
* **No Real-Time Processing**: MapReduce processes data in batches, making it unsuitable for real-time or streaming applications.

## 🏁 Conclusion

MapReduce revolutionized large-scale data processing by providing a simple, powerful model that hid distributed system complexities. While Google has since moved to advanced systems, MapReduce influenced frameworks like Hadoop and Spark, shaping modern data processing.
