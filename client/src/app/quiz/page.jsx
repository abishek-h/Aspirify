"use client";
import React from "react";
import { useState } from "react";
import SyncLoader from "react-spinners/SyncLoader";
import ques from "../jsons/ques.json";
import styles from "./quiz.module.css";
import axios from "axios";
import { useRouter } from "next/navigation";

const Quiz = () => {
  const [loading, setloading] = useState(false);
  const [arr, setarr] = useState([]);
  const que = ques.questions;
  const answers = ques.answers;
  const [qno, setqno] = useState(0);
  const [topping, settopping] = useState(null);
  const [form, setform] = useState(true);
  const [quiz, setquiz] = useState(true);
  const [results, setresults] = useState();
  const router = useRouter();
  const url = "https://aspirify-backend.onrender.com/prediction";
  const retake = () => {
    setarr([]);
    setqno(0);
    settopping(null);
    setform(true);
    setquiz(true);
  };
  const next = () => {
    if (qno <= 16) {
      if (qno < 16) {
        setqno(qno + 1);
      }
      topping == null ? setarr([...arr, 1]) : setarr([...arr, topping]);
      if (qno == 16) {
        setform(false);
      }
    }
    settopping(null);
  };
  const post = async () => {
    setloading(true);
    await axios
      .post(url, {
        arr,
      })
      .then((res) => setresults(res.data.role));
    setquiz(false);
    setloading(false);
  };
  const reset = () => {
    setqno(0);
    setarr([]);
    settopping(null);
  };
  return (
    <div className={styles.king}>
      {quiz && (
        <div className={styles.subking}>
          <div>
            <img
              className={styles.quizimg}
              src={form ? "./qui.webp" : "./qdone.webp"}
            ></img>
          </div>
          <div className={styles.form}>
            {form && (
              <>
                <div className={styles.ques}>
                  <span>Question</span>
                  <span>{qno + 1}</span>
                  <span>/17</span>
                </div>
                <div className={styles.question}>
                  <span className={styles.q}>{que[qno].question}</span>
                </div>
                <div className={styles.answer}>
                  {answers?.map((dat, i) => {
                    return (
                      <div className={styles.options}>
                        <label className={styles.label} for={dat.ans}>
                          <input
                            type="radio"
                            value={dat.id}
                            id={dat.ans}
                            name="topping"
                            checked={topping === dat.id}
                            onChange={(e) => {
                              settopping(parseInt(e.target.value));
                            }}
                            className={styles.input}
                          />
                          <div className={styles.answ}>{dat.ans}</div>
                        </label>
                      </div>
                    );
                  })}
                </div>
                <div className={styles.buttons}>
                  {qno > 0 && (
                    <button className={styles.bleft} onClick={reset}>
                      Reset
                    </button>
                  )}
                  <button className={styles.bright} onClick={next}>
                    {qno === 16 ? "Submit" : "Next"}
                  </button>
                </div>
              </>
            )}
            {!form && !loading && (
              <div className={styles.redirect}>
                <div className={styles.message}>
                  Congratulations on completing the quiz!
                </div>
                <div>
                  <button className={styles.bleft} onClick={post} type="button">
                    Results
                  </button>
                </div>
              </div>
            )}
            {loading && (
              <div className={styles.loa}>
                <SyncLoader color="#da3a00" loading={loading} size={18} />
              </div>
            )}
          </div>
        </div>
      )}
      {!quiz && (
        <div className={styles.result}>
          <div className={styles.suggestionl}>
            <div className={styles.dia}>
              Consequently, these are the{" "}
              <span style={{ color: "#da3a00" }}>positions</span> that, given
              your skills, are best suited for you.
            </div>
            {results?.map((data) => {
              return (
                <div
                  className={styles.options}
                  style={{ color: "#da3a00", fontSize: "1.2rem" }}
                >
                  {data}
                </div>
              );
            })}
            <div className={styles.botores}>
              <button className={styles.bleft} onClick={retake}>
                Retake Quiz
              </button>
              <button
                className={styles.bright}
                onClick={() => {
                  router.push("/");
                }}
              >
                Exit
              </button>
            </div>
          </div>
          <div>
            <img className={styles.sugimg} src="./suggestions.webp"></img>
          </div>
        </div>
      )}
    </div>
  );
};

export default Quiz;
