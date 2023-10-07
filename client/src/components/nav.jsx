"use client";
import React from "react";
import Link from "next/link";
import { useState } from "react";

const nav = () => {
  const [showserv, setshowserv] = useState(false);
  const show = () => {
    setshowserv(true);
  };
  const notshow = () => {
    setTimeout(() => {
      setshowserv(false);
    }, 1500);
  };
  return (
    <div className="navbar flex justify-between min-w-[960px] ">
      <div className="nav-left pl-[2.6rem] pt-8 ">
        <Link href="/">
          <div className="title text-[3.2rem]">Aspirify</div>
          <div className="sutitle text-[1.1rem]">Career Compass</div>
        </Link>
      </div>
      <div className="navigations flex w-[30rem] justify-around text-[1.2rem] pt-5 items-center pr-4 ">
        <Link href="/">
          <div className="nav-items text-black hover:bg-[#da3a00] hover:text-white p-2">
            Home
          </div>
        </Link>
        <div
          className="nav-items relative"
          onMouseEnter={show}
          onMouseLeave={notshow}
        >
          <Link
            href="/service"
            className="text-black hover:bg-[#da3a00] hover:text-white p-2"
          >
            Services
          </Link>
          <div>
            {showserv && (
              <div className="service-options text-[1.2rem] border-black border-[1px] absolute mt-4 right-[-1rem] top-[1rem] p-2 flex flex-col h-[6.5rem] justify-between bg-white rounded-[5px]">
                <Link href="/quiz">
                  <div className="sr-op hover:text-[#da3a00] cursor-pointer">
                    Quiz
                  </div>
                </Link>
                <Link href="/roadmap">
                  <div className="sr-op hover:text-[#da3a00] cursor-pointer">
                    Roadmap
                  </div>
                </Link>
                <Link href="/degree">
                  <div className="sr-op hover:text-[#da3a00] cursor-pointer">
                    Degree
                  </div>
                </Link>
              </div>
            )}
          </div>
        </div>
        <div className="nav-ite">
          <Link
            href="/login"
            className="text-black hover:bg-[#da3a00] hover:text-white p-2"
          >
            log in/sign up
          </Link>
        </div>
      </div>
    </div>
  );
};

export default nav;
