"use client";
import React from "react";
import { useState } from "react";
import Link from "next/link";

const bachelor = () => {
  const [toggle, settoggle] = useState(null);
  const toggled = (i) => {
    if (toggle === i) {
      return settoggle(null);
    }
    settoggle(i);
  };
  const degree = [
    {
      name: "B.Tech in Computer Science and Engineering (CSE)",
      ans: "This is the most traditional computer science degree. It covers a broad range of topics, including programming languages, data structures and algorithms, computer architecture, operating systems, and software engineering.",
    },
    {
      name: "B.Tech in Information Technology (IT)",
      ans: "This degree is more applied than the B.Tech in CSE. It focuses on the use of computer technology to solve real-world problems. Students in this program typically take courses in database management, web development, and network administration.",
    },
    {
      name: "B.Tech in Software Engineering (SE)",
      ans: "This is a specialized degree that prepares students to design, develop, and test software systems. Students in this program typically take courses in software engineering principles, software design methodologies, and software testing.",
    },
    {
      name: "B.Tech in Data Science (DS)",
      ans: "This is an interdisciplinary degree that combines computer science with statistics and mathematics. Students in this program learn how to collect, clean, and analyze data to extract meaningful insights.",
    },
    {
      name: "B.Tech in Cybersecurity (CS)",
      ans: "This is a specialized degree that prepares students to protect computer systems and networks from cyberattacks. Students in this program typically take courses in cryptography, network security, and security analytics.",
    },
    {
      name: "B.Tech in Artificial Intelligence (AI)",
      ans: "This is a specialized degree that prepares students to develop intelligent systems. Students in this program typically take courses in machine learning, natural language processing, and computer vision.",
    },
    {
      name: "B.Tech in Robotics",
      ans: "This is a specialized degree that prepares students to design, build, and control robots. Students in this program typically take courses in robotics mechanics, electronics, and software engineering.",
    },
    {
      name: "B.Tech in Bioinformatics",
      ans: "This is an interdisciplinary degree that combines computer science with biology. Students in this program learn how to use computer technology to solve biological problems, such as drug discovery and disease diagnosis.",
    },
    {
      name: "B.Tech in Computer Networking",
      ans: "This degree focuses on the design, implementation, and management of computer networks. Students in this program typically take courses in network protocols, network security, and network management.",
    },
    {
      name: "B.Tech in Computer Graphics",
      ans: "This degree focuses on the creation and manipulation of digital images and videos. Students in this program typically take courses in computer graphics fundamentals, 3D modeling, and animation.",
    },
    {
      name: "B.Tech in Computer Games and Animation",
      ans: "This degree focuses on the design, development, and production of computer games and animations. Students in this program typically take courses in game design, game programming, and game animation.",
    },
    {
      name: "B.Tech in Multimedia Technology",
      ans: "This degree focuses on the creation and delivery of multimedia content, such as digital video, audio, and images. Students in this program typically take courses in multimedia authoring, video editing, and audio production.",
    },
    {
      name: "B.Tech in Web Technology",
      ans: "This degree focuses on the design, development, and management of web applications and websites. Students in this program typically take courses in web programming, web design, and web database management.",
    },
  ];
  return (
    <div className="w-[97vw] mr-auto ml-auto p-[1rem] min-w-[1100px] mt-[1.5rem] flex h-[784px] align-top">
      <div>
        <Link href="/degree">
          <button className="w-[5rem] mr-[7rem] border-solid border-[#da3a00] border-[1px] text-[1.3rem] pr-[.75rem] pl-[.75rem] pt-[.4rem] pb-[.4rem] text-[#da3a00] transition-all hover:scale-[1.1] duration-200">
            Back
          </button>
        </Link>
      </div>
      <div className="flex flex-col mr-[4rem]">
        <div className="text-[1.5rem] text-[#da3a00] mb-[1rem]">
          Bachelor's Degrees
        </div>
        <div className="flex flex-col gap-3 ">
          {degree.map((item, i) => {
            return (
              <div
                className="shadow-[3.5px_3.5px_rgba(218,58,0,0.85)] border-solid border-[1px] border-[#da3a00] min-w-[35rem] p-[.5rem] cursor-pointer text-[1rem] w-[40vw]"
                onClick={() => {
                  toggled(i);
                }}
              >
                <div className="flex w-[39vw] justify-between min-w-[34rem]">
                  <div className="font-medium">{item.name}</div>
                  <div>{toggle === i ? "-" : "+"}</div>
                </div>
                <div
                  className={
                    toggle === i
                      ? "w-[39vw] min-w-[34rem] transition-[height,opacity] h-[5rem] duration-250 opacity-100 pt-[.3rem] text-justify pb-[.5rem] pr-[.1rem] pl-[.1rem]"
                      : "w-[39vw] min-w-[34rem] h-0 overflow-hidden transition-[height,opacity] duration-200 opacity-0 text-justify"
                  }
                >
                  {item.ans}
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <div className="">
        <img
          src="/bach.png"
          className="w-[100%] min-w-[300px] h-[745px] object-cover "
        ></img>
      </div>
    </div>
  );
};

export default bachelor;
