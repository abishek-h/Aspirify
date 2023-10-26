"use client";
import React from "react";
import { useState } from "react";

const Roadmap = () => {
  const [roadmap, setroadmap] = useState(true);
  const roles = [
    { name: "frontend", key: "frontend" },
    { name: "backend", key: "backend" },
    { name: "full stack", key: "fullstack" },
    { name: "devops", key: "devops" },
    { name: "software architect", key: "softwarearchitect" },
    { name: "android", key: "android" },
    { name: "AI data scientist", key: "ai_data" },
    { name: "Block Chain", key: "blockchain" },
  ];
  const skills = [
    { name: "Angular", key: "angular" },
    { name: "Java", key: "java" },
    { name: "JavaScript", key: "js" },
    { name: "Node.js", key: "nodejs" },
    { name: "React.js", key: "react" },
    { name: "Sql", key: "sql" },
    { name: "Python", key: "python" },
  ];
  const [rd, setrd] = useState([]);
  return (
    <div>
      {roadmap && (
        <div
          style={{
            background: `url("17.webp")`,
            backgroundSize: "contain",
            backgroundRepeat: "no-repeat",
            backgroundPosition: "center",
            backgroundPositionY: -60,
          }}
          className="h-[87vh] flex flex-col min-w-[970px]"
        >
          <div className="title text-[#da3a00] text-[2.5rem] text-center font-extralight  border-b-[1.5px] border-[#da3a00] pb-[1rem] w-[90%] ml-auto mr-auto mt-[5.5rem]">
            Roadmap
          </div>
          <div className="w-[90%] ml-auto mr-auto">
            <div className="text-[#da3a00] mt-[0.3rem] text-[1.5rem]  text-center inline-block">
              Role-based
            </div>
            <div className="ml-auto mr-auto flex min-h-[17vh] gap-x-[1rem] mt-[0.5rem] w-[80%] flex-wrap justify-center">
              {roles.map((role) => {
                return (
                  <button
                    onClick={() => {
                      setroadmap(false);
                      setrd(role.key);
                    }}
                    className="bg-white m-[0.5rem] cursor-pointer border-[#da3a00] border-solid border-[1px] shadow-[3.5px_3.5px_rgba(218,58,0,0.85)] flex justify-center p-[1rem] items-center text-[1.5rem] font-light h-[3rem] text-[#da3a00] transition-all hover:scale-[1.1] duration-200"
                  >
                    {role.name}
                  </button>
                );
              })}
            </div>
          </div>
          <div className="border-t-[1.5px] border-solid border-[#da3a00] w-[70vw] ml-auto mr-auto mt-[1rem] min-w-[846px]">
            <div className="text-[#da3a00] text-center mt-[0.3rem] text-[1.5rem]  inline-block">
              Skill-based
            </div>
            <div className="ml-auto mr-auto flex min-h-[17vh] gap-x-[1rem] mt-[0.5rem] w-[80%] flex-wrap justify-center">
              {skills.map((skill) => {
                return (
                  <button
                    onClick={() => {
                      setroadmap(false);
                      setrd(skill.key);
                    }}
                    className="bg-white m-[0.5rem] cursor-pointer border-[#da3a00] border-solid border-[1px] shadow-[3.5px_3.5px_rgba(218,58,0,0.85)] flex justify-center p-[1rem] items-center text-[1.5rem] font-light h-[3rem] text-[#da3a00] transition-all hover:scale-[1.1] duration-200"
                  >
                    {skill.name}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}
      {!roadmap && (
        <div>
          <button
            className="text-[1.85rem] mb-[1rem] text-[#da3a00] ml-[2.5rem] mt-[1rem] border-[1px] pt-[.4rem] pl-[.6rem] pr-[.6rem] pb-[.4rem] border-[#da3a00] font-light transition-all hover:scale-[1.1] duration-150"
            onClick={() => {
              setroadmap(true);
            }}
          >
            &#60; Back
          </button>
          <div>
            <img src={`${rd}.webp`} className="w-[65vw] ml-auto mr-auto"></img>
          </div>
        </div>
      )}
    </div>
  );
};

export default Roadmap;
