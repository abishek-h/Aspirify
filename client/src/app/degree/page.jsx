import React from "react";
import Link from "next/link";

const Degree = () => {
  const why = [
    {
      title: "Increased earning potential.",
      text: ": Graduates of bachelor's and master's programs typically earn more money than those with only a high school diploma.",
    },
    {
      title: "Better job prospects.",
      text: ": Graduates of bachelor's and master's programs are in high demand in many industries.",
    },
    {
      title: "Greater job satisfaction.",
      text: ": Graduates of bachelor's and master's programs often report higher levels of job satisfaction than those with only a high school diploma.",
    },
    {
      title: "Improved critical thinking and problem-solving skills.",
      text: ": Bachelor's and master's programs teach students how to think critically and solve problems effectively.",
    },
    {
      title: "Greater self-confidence.",
      text: ": Completing a bachelor's or master's degree can give students a sense of accomplishment and boost their self-confidence.",
    },
  ];
  return (
    <div
      className="h-[85vh] mt-[.5rem]"
      style={{
        background: `url("degre.webp")`,
        backgroundPositionY: -500,
        backgroundPositionX: -100,
      }}
    >
      <div className="w-[96vw] ml-auto mr-auto min-w-[1100px] pr-[2.5rem] pl-[2.5rem] pt-[1rem]">
        <div className="text-[1.3rem]">
          Welcome to our{" "}
          <span className="text-[#da3a00] font-medium">master's</span> and{" "}
          <span className="text-[#da3a00] font-medium">bachelor's</span> degree
          page! You may learn more about a variety of{" "}
          <span className="text-[#da3a00] font-medium">computer science</span>{" "}
          degree programs here.
        </div>
        <div className="mt-[0.8rem]">
          <Link href="/degree/bachelor">
            <button className="text-[1.35rem] border-solid border-[1px] border-[#da3a00] h-[4.5rem] w-[10rem] p-[.5rem] text-justify bg-white text-[#da3a00] mr-[1.5rem] shadow-[3.5px_3.5px_rgba(218,58,0,0.85)] transition-all hover:scale-[1.1] duration-200">
              Bachelor's Degrees
            </button>
          </Link>
          <Link href="/degree/master">
            <button className="text-[1.35rem] border-solid border-[1px] border-[#da3a00] h-[4.5rem] w-[10rem] p-[.5rem] text-justify bg-white text-[#da3a00] shadow-[3.5px_3.5px_rgba(218,58,0,0.85)] transition-all hover:scale-[1.1] duration-200">
              Master's Degrees
            </button>
          </Link>
        </div>
        <div>
          <div className="text-[1.5rem] mt-[1.5rem] text-[#da3a00] font-medium">
            Bachelor's Degree
          </div>
          <div className="text-[1.25rem] mt-[.5rem]">
            A bachelor's degree is a four-year undergraduate degree that is the
            first step in many careers. Bachelor's programs provide students
            with a broad foundation of knowledge in their chosen field of study.
          </div>
        </div>
        <div>
          <div className="text-[1.5rem] text-[#da3a00] font-medium mt-[1rem]">
            Master's Degree
          </div>
          <div className="text-[1.25rem] mt-[.5rem]">
            A master's degree is a two-year graduate degree that provides
            students with specialized knowledge and skills in a particular
            field. Master's programs are typically designed for students who
            already have a bachelor's degree and want to advance their education
            and career.
          </div>
        </div>
        <div className="text-[1.5rem] font-medium mt-[1rem]">
          Why get a bachelor's or master's degree?
        </div>
        <div className="text-[1.25rem] mt-[.5rem]">
          There are many benefits to getting a bachelor's or master's degree.
          Some of the most common benefits include :
        </div>
        <div className="pl-[1.5rem] mt-[.5rem]">
          {why.map((data) => {
            return (
              <div className="flex items-center">
                <div className="text-[1.8rem] mr-[.5rem]">&#10526;</div>
                <div className="flex items-center">
                  <div className="text-[1.25rem] font-medium">{data.title}</div>
                  {/* <div className="text-[1.25rem]">{data.text}</div> */}
                </div>
              </div>
            );
          })}
        </div>
        <div className="text-[1.5rem] font-medium mt-[1rem]">
          How to choose a bachelor's or master's degree program
        </div>
        <div className="text-[1.25rem] mt-[.5rem] text-justify">
          The curriculum of the program, the reputation of the institution, your
          interests, your ambitions, your financial condition, and the school's
          location should all be taken into account when selecting a bachelor's
          or master's degree program. You may choose the best kind of program
          and school for you based on your interests and objectives. The amount
          of tuition you can pay and whether you need to apply for grants or
          scholarships will depend on your financial status. What you will be
          studying and the kind of research possibilities that are accessible
          will be made clear to you by the program's curriculum. Your
          understanding of the school's standing and the achievement of its
          alumni may be gained from its reputation. Where you live while
          enrolled in school depends on the location of the institution. It is
          important to weigh all of these factors carefully when choosing a
          bachelor's or master's degree program. The right program can set you
          up for success in your chosen career.
        </div>
      </div>
    </div>
  );
};

export default Degree;
