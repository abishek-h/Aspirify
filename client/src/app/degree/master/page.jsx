import React from "react";
import Link from "next/link";

const masters = () => {
  const degree = [
    "Master of Science in Computer Science (MS in CS)",
    "Master of Science in Software Engineering (MS in SE)",
    "Master of Science in Data Science (MS in DS)",
    "Master of Science in Information Technology (MS in IT)",
    "Master of Science in Cybersecurity (MS in CS)",
    "Master of Science in Artificial Intelligence (MS in AI)",
    "Master of Science in Robotics (MS in Robotics)",
    "Master of Science in Bioinformatics (MS in Bioinformatics)",
    "Master of Science in Computer Networking (MS in CN)",
    "Master of Science in Computer Graphics (MS in CG)",
    "Master of Science in Computer Games and Animation (MS in CG&A)",
    "Master of Science in Multimedia Technology (MS in MT)",
    "Master of Science in Web Technology (MS in WT)",
  ];
  return (
    <div className="w-[97vw] mr-auto ml-auto p-[1rem] min-w-[1100px] mt-[1.5rem] flex h-[784px]">
      <div>
        <Link href="/degree">
          <button className="w-[5rem] mr-[7rem] border-solid border-[#da3a00] border-[1px] text-[1.3rem] pr-[.75rem] pl-[.75rem] pt-[.4rem] pb-[.4rem] text-[#da3a00] transition-all hover:scale-[1.1] duration-200">
            Back
          </button>
        </Link>
      </div>
      <div className="flex flex-col mr-[4rem]">
        <div className="text-[1.5rem] text-[#da3a00] mb-[1rem]">
          Master's Degrees
        </div>
        <div className="flex flex-col gap-3 ">
          {degree.map((item) => {
            return (
              <div className="shadow-[3.5px_3.5px_rgba(218,58,0,0.85)] border-solid border-[1px] border-[#da3a00] min-w-[35rem] p-[.5rem] cursor-pointer text-[1rem] w-[40vw] transition-all hover:scale-[1.05] duration-200">
                {item}
              </div>
            );
          })}
        </div>
      </div>
      <div>
        <img
          src="/mast.png"
          className="w-[100%] min-w-[300px] h-[745px] object-cover"
        ></img>
      </div>
    </div>
  );
};

export default masters;
