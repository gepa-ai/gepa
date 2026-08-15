---
template: workbench.html
title: GEPA Workbench
description: Observability and steerability for your optimize_anything runs.
search:
  exclude: true
hide:
  - navigation
  - toc
---

<div class="wb-page">

<!-- ================= Hero ================= -->

<div class="wb-hero">
  <h1>Inside GEPA Workbench</h1>
  <p class="wb-hero-sub">Adding observability and steerability to your optimize_anything runs.</p>
  <div class="wb-cta-row">
    <a class="wb-btn" href="https://forms.gle/GCD4PmjdLrKUB8Eg7" target="_blank" rel="noopener">Join the study</a>
  </div>
</div>

<!-- ================= What is GEPA Workbench ================= -->

<div class="wb-section wb-intro">
  <h3>What is GEPA Workbench?</h3>
  <p>GEPA Workbench is an interactive platform to observe, inspect, and steer your optimize_anything runs. The workbench lets you inspect proposed candidates, evaluations, and reflections. You can also step in at any point to steer the search with your feedback, or have GEPA Agent diagnose and propose fixes for potential issues.</p>
</div>

<!-- ================= Entry cards ================= -->

<div class="wb-section">
  <h3>Starting an optimization run</h3>
  <p>You can submit optimize_anything runs as jobs, configure and launch new runs directly in your browser, or upload a finished run to inspect it post-completion.</p>
  <div class="wb wb-entries">
    <div class="wb-entry">
      <div class="ico">❯_</div>
      <h3>Submit a job</h3>
      <p>Launch a run using the submit_job call and observe it live in the workbench.</p>
      <span class="snippet"><span class="k">from</span> gepa.dashboard <span class="k">import</span> <span class="f">submit_job</span></span>
    </div>
    <div class="wb-entry">
      <div class="ico">✦</div>
      <h3>Configure &amp; launch</h3>
      <p>Set up and launch a new optimization run in the browser.</p>
      <span class="cta">+ New run</span>
    </div>
    <div class="wb-entry">
      <div class="ico">↑</div>
      <h3>Inspect a finished run</h3>
      <p>Upload your gepa_state.bin to replay a completed run at full fidelity.</p>
      <span class="cta">Upload a run</span>
    </div>
  </div>
</div>

<!-- ================= Observability ================= -->

<div class="wb-section">
  <h3>Observe your optimization runs</h3>
  <p>Observe and inspect overall progress, individual candidates, training and evaluation outcomes, and model reflections.</p>
  <div class="wb wb-scroll">
    <div class="wb-window wb-obs">
      <div class="wb-chrome"><i></i><i></i><i></i><span>GEPA Workbench · circle-packing-multi</span></div>
      <div class="wb-hdr">
        <span class="wb-back">← Runs</span>
        <span class="wb-vdiv"></span>
        <span class="wb-logo"><img src="../static/img/gepa_logo.svg" alt="GEPA">GEPA</span>
        <span class="wb-runsel">circle-packing-multi<b>▾</b></span>
        <span class="wb-badge"><span class="wb-dot"></span>In progress</span>
        <div class="wb-stats">
          <div class="wb-stat"><span>Agg score</span><b class="teal">0.87 <i class="up">(+0.14)</i></b></div>
          <div class="wb-stat"><span>Calls</span><b>418 / 600</b></div>
          <div class="wb-stat"><span>ETA</span><b>~4m</b></div>
        </div>
        <div class="wb-actions"><span>Pause</span><span>Stop</span><span class="tealb">Branch</span></div>
      </div>
      <div class="wb-body">
        <aside class="wb-explorer">
          <div class="wb-exp-head">Candidates</div>
          <div class="wb-seg"><span class="on">List</span><span>Lineage</span></div>
          <div class="wb-cands">
            <div class="wb-cand">Candidate 0 <span class="mk sc">0.73</span></div>
            <div class="wb-cand">Candidate 1 <span class="mk sc">0.75</span></div>
            <div class="wb-cand">Candidate 2 <span class="mk sc">0.71</span></div>
            <div class="wb-cand">Candidate 3 <span class="mk sc">0.78</span></div>
            <div class="wb-cand">Candidate 4 <span class="mk par">◆</span><span class="sc">0.80</span></div>
            <div class="wb-cand">Candidate 5 <span class="mk sc">0.74</span></div>
            <div class="wb-cand">Candidate 6 <span class="mk par">◆</span><span class="sc">0.84</span></div>
            <div class="wb-cand on">Candidate 7 <span class="mk star">★</span><span class="sc">0.87</span></div>
          </div>
        </aside>
        <div class="wb-main">
          <div class="wb-tabs">
            <span class="wb-tab wb-tab-ov">Overview</span>
            <span class="wb-tab wb-tab-ev">Evaluation</span>
            <span class="wb-tab wb-tab-rf">Reflections</span>
          </div>
          <div class="wb-panes">
            <div class="wb-pane wb-pane-ov">
              <div class="wb-card wb-banner">
                <span class="wb-dot"></span><b>Improving</b>
                <span>Best score improved to 0.87 at iteration 12, +0.14 over the seed.</span>
                <span class="wb-analyze">✦ Analyze</span>
              </div>
              <div class="wb-card wb-hero-card">
                <div class="wb-hc-left">
                  <div class="wb-hc-label">Current best</div>
                  <div class="wb-hc-score"><i>0.75</i><i>0.80</i><i>0.87</i><em>▲ +0.14</em></div>
                  <div class="wb-hc-sub">Candidate 7 <span class="chip">+19.2%</span></div>
                </div>
                <div class="wb-hc-mid">
                  <div class="wb-hc-comp">program <span>212 tok</span></div>
                  <div class="wb-hc-prev">def main(n, timeout, best): pts, r = anneal(grid_layout(n), steps_for(n)) …</div>
                </div>
                <div class="wb-hc-actions"><span class="ghost">Copy</span><span class="solid">Open ›</span></div>
              </div>
              <div class="wb-card wb-chart-card wb-chart">
                <div class="wb-chart-head">
                  <b>Score over time</b>
                  <div class="ctl"><span>Show notes</span><span>Pop out</span></div>
                </div>
                <svg viewBox="0 0 760 190" aria-label="Score over metric calls">
                  <line class="grid" x1="46" y1="142" x2="742" y2="142"></line>
                  <line class="grid" x1="46" y1="98" x2="742" y2="98"></line>
                  <line class="grid" x1="46" y1="54" x2="742" y2="54"></line>
                  <line class="axis" x1="46" y1="164" x2="742" y2="164"></line>
                  <line class="axis" x1="46" y1="10" x2="46" y2="164"></line>
                  <text class="tick" x="40" y="145" text-anchor="end">0.65</text>
                  <text class="tick" x="40" y="101" text-anchor="end">0.75</text>
                  <text class="tick" x="40" y="57" text-anchor="end">0.85</text>
                  <text class="tick" x="46" y="176" text-anchor="middle">0</text>
                  <text class="tick" x="220" y="176" text-anchor="middle">150</text>
                  <text class="tick" x="394" y="176" text-anchor="middle">300</text>
                  <text class="tick" x="568" y="176" text-anchor="middle">450</text>
                  <text class="tick" x="742" y="176" text-anchor="middle">600</text>
                  <text class="lbl" x="394" y="188" text-anchor="middle">Metric calls</text>
                  <line class="ref-seed" x1="46" y1="107" x2="742" y2="107"></line>
                  <line class="ref-best" x1="46" y1="45" x2="742" y2="45"></line>
                  <line class="ref-pareto" x1="46" y1="28" x2="742" y2="28"></line>
                  <rect class="ychip" x="6" y="100.5" width="34" height="13" rx="3"></rect>
                  <text class="ychip-txt" x="23" y="110" text-anchor="middle">0.73</text>
                  <rect class="ychip ychip-best" x="6" y="38.5" width="34" height="13" rx="3"></rect>
                  <text class="ychip-txt ychip-txt-best" x="23" y="48" text-anchor="middle">0.87</text>
                  <path class="traj" pathLength="1" d="M81,107 L150,98 L220,116 L278,85 L353,76 L417,102 L475,58 L531,45"></path>
                  <circle class="cdot p1" cx="81" cy="107" r="3.5"></circle>
                  <circle class="cdot p2" cx="150" cy="98" r="3.5"></circle>
                  <circle class="cdot p3" cx="220" cy="116" r="3.5"></circle>
                  <circle class="cdot p4" cx="278" cy="85" r="3.5"></circle>
                  <circle class="cdot p5" cx="353" cy="76" r="3.5"></circle>
                  <circle class="cdot p6" cx="417" cy="102" r="3.5"></circle>
                  <circle class="cdot p7" cx="475" cy="58" r="3.5"></circle>
                  <circle class="chalo p8" cx="531" cy="45" r="9"></circle>
                  <circle class="cbest p8" cx="531" cy="45" r="5"></circle>
                </svg>
                <div class="wb-legend">
                  <span class="lg-seed">seed 0.73</span>
                  <span class="lg-best">best 0.87</span>
                  <span class="lg-par">pareto 0.91</span>
                </div>
              </div>
            </div>
            <div class="wb-pane wb-pane-ev">
              <div class="wb-ev-toolbar">
                <b>10 tasks</b>
                <div class="wb-seg"><span class="on">Fields &amp; Scores</span><span>Fields Only</span><span>Scores Only</span></div>
              </div>
              <div class="wb-table">
                <div class="wb-tr wb-th"><div>#</div><div>n_circles</div><div>sum_radii</div><div>circles</div><div>candidates 0–7</div></div>
                <div class="wb-tr"><div class="mono">1</div><div class="mono">1</div><div class="mono strong">0.500</div><div><svg class="wb-pk" viewBox="0 0 20 20"><rect x="0.5" y="0.5" width="19" height="19"></rect><circle cx="10" cy="10" r="9"></circle></svg></div><div><span class="wb-strip"><i class="s5 k1"></i><i class="s5 k2"></i><i class="s5 k3"></i><i class="s5 k4"></i><i class="s5 k5"></i><i class="s5 k6"></i><i class="s5 k7"></i><i class="s5 k8"></i></span></div></div>
                <div class="wb-tr"><div class="mono">2</div><div class="mono">2</div><div class="mono strong">0.586</div><div><svg class="wb-pk" viewBox="0 0 20 20"><rect x="0.5" y="0.5" width="19" height="19"></rect><circle cx="5.9" cy="14.1" r="5.6"></circle><circle cx="14.1" cy="5.9" r="5.6"></circle></svg></div><div><span class="wb-strip"><i class="s4 k1"></i><i class="s4 k2"></i><i class="s4 k3"></i><i class="s5 k4"></i><i class="s5 k5"></i><i class="s4 k6"></i><i class="s5 k7"></i><i class="s5 k8"></i></span></div></div>
                <div class="wb-tr"><div class="mono">3</div><div class="mono">3</div><div class="mono strong">0.663</div><div><svg class="wb-pk" viewBox="0 0 20 20"><rect x="0.5" y="0.5" width="19" height="19"></rect><circle cx="5" cy="5" r="4.4"></circle><circle cx="15" cy="5" r="4.4"></circle><circle cx="10" cy="14" r="4.4"></circle></svg></div><div><span class="wb-strip"><i class="s3 k1"></i><i class="s4 k2"></i><i class="s3 k3"></i><i class="s4 k4"></i><i class="s4 k5"></i><i class="s4 k6"></i><i class="s5 k7"></i><i class="s5 k8"></i></span></div></div>
                <div class="wb-tr"><div class="mono">4</div><div class="mono">4</div><div class="mono strong">0.714</div><div><svg class="wb-pk" viewBox="0 0 20 20"><rect x="0.5" y="0.5" width="19" height="19"></rect><circle cx="5.5" cy="5.5" r="4.4"></circle><circle cx="14.5" cy="5.5" r="4.4"></circle><circle cx="5.5" cy="14.5" r="4.4"></circle><circle cx="14.5" cy="14.5" r="4.4"></circle></svg></div><div><span class="wb-strip"><i class="s3 k1"></i><i class="s3 k2"></i><i class="s3 k3"></i><i class="s4 k4"></i><i class="s4 k5"></i><i class="s3 k6"></i><i class="s4 k7"></i><i class="s5 k8"></i></span></div></div>
                <div class="wb-tr"><div class="mono">5</div><div class="mono">5</div><div class="mono strong">0.741</div><div><svg class="wb-pk" viewBox="0 0 20 20"><rect x="0.5" y="0.5" width="19" height="19"></rect><circle cx="4.5" cy="4.5" r="3.6"></circle><circle cx="15.5" cy="4.5" r="3.6"></circle><circle cx="4.5" cy="15.5" r="3.6"></circle><circle cx="15.5" cy="15.5" r="3.6"></circle><circle cx="10" cy="10" r="3.9"></circle></svg></div><div><span class="wb-strip"><i class="s2 k1"></i><i class="s3 k2"></i><i class="s2 k3"></i><i class="s3 k4"></i><i class="s4 k5"></i><i class="s3 k6"></i><i class="s4 k7"></i><i class="s5 k8"></i></span></div></div>
                <div class="wb-tr"><div class="mono">6</div><div class="mono">6</div><div class="mono strong">0.766</div><div><svg class="wb-pk" viewBox="0 0 20 20"><rect x="0.5" y="0.5" width="19" height="19"></rect><circle cx="4.3" cy="6" r="2.8"></circle><circle cx="10" cy="6" r="2.8"></circle><circle cx="15.7" cy="6" r="2.8"></circle><circle cx="4.3" cy="14" r="2.8"></circle><circle cx="10" cy="14" r="2.8"></circle><circle cx="15.7" cy="14" r="2.8"></circle></svg></div><div><span class="wb-strip"><i class="s2 k1"></i><i class="s2 k2"></i><i class="s2 k3"></i><i class="s3 k4"></i><i class="s3 k5"></i><i class="s3 k6"></i><i class="s4 k7"></i><i class="s4 k8"></i></span></div></div>
                <div class="wb-tr"><div class="mono">7</div><div class="mono">7</div><div class="mono strong">0.788</div><div><svg class="wb-pk" viewBox="0 0 20 20"><rect x="0.5" y="0.5" width="19" height="19"></rect><circle cx="10" cy="10" r="3"></circle><circle cx="10" cy="3.5" r="2.9"></circle><circle cx="15.6" cy="6.8" r="2.9"></circle><circle cx="15.6" cy="13.2" r="2.9"></circle><circle cx="10" cy="16.5" r="2.9"></circle><circle cx="4.4" cy="13.2" r="2.9"></circle><circle cx="4.4" cy="6.8" r="2.9"></circle></svg></div><div><span class="wb-strip"><i class="s2 k1"></i><i class="s2 k2"></i><i class="s1 k3"></i><i class="s3 k4"></i><i class="s3 k5"></i><i class="s2 k6"></i><i class="s4 k7"></i><i class="s4 k8"></i></span></div></div>
                <div class="wb-tr"><div class="mono">8</div><div class="mono">8</div><div class="mono strong">0.803</div><div><svg class="wb-pk" viewBox="0 0 20 20"><rect x="0.5" y="0.5" width="19" height="19"></rect><circle cx="4" cy="4" r="2.6"></circle><circle cx="10" cy="4" r="2.6"></circle><circle cx="16" cy="4" r="2.6"></circle><circle cx="7" cy="10" r="2.6"></circle><circle cx="13" cy="10" r="2.6"></circle><circle cx="4" cy="16" r="2.6"></circle><circle cx="10" cy="16" r="2.6"></circle><circle cx="16" cy="16" r="2.6"></circle></svg></div><div><span class="wb-strip"><i class="s1 k1"></i><i class="s2 k2"></i><i class="s1 k3"></i><i class="s2 k4"></i><i class="s3 k5"></i><i class="s2 k6"></i><i class="s3 k7"></i><i class="s4 k8"></i></span></div></div>
                <div class="wb-tr"><div class="mono">9</div><div class="mono">9</div><div class="mono strong">0.812</div><div><svg class="wb-pk" viewBox="0 0 20 20"><rect x="0.5" y="0.5" width="19" height="19"></rect><circle cx="4" cy="4" r="2.9"></circle><circle cx="10" cy="4" r="2.9"></circle><circle cx="16" cy="4" r="2.9"></circle><circle cx="4" cy="10" r="2.9"></circle><circle cx="10" cy="10" r="2.9"></circle><circle cx="16" cy="10" r="2.9"></circle><circle cx="4" cy="16" r="2.9"></circle><circle cx="10" cy="16" r="2.9"></circle><circle cx="16" cy="16" r="2.9"></circle></svg></div><div><span class="wb-strip"><i class="s1 k1"></i><i class="s1 k2"></i><i class="s1 k3"></i><i class="s2 k4"></i><i class="s2 k5"></i><i class="s2 k6"></i><i class="s3 k7"></i><i class="s4 k8"></i></span></div></div>
                <div class="wb-tr"><div class="mono">10</div><div class="mono">10</div><div class="mono strong">0.820</div><div><svg class="wb-pk" viewBox="0 0 20 20"><rect x="0.5" y="0.5" width="19" height="19"></rect><circle cx="4.6" cy="3.8" r="2.3"></circle><circle cx="10" cy="3.8" r="2.3"></circle><circle cx="15.4" cy="3.8" r="2.3"></circle><circle cx="3.2" cy="10" r="2.3"></circle><circle cx="7.7" cy="10" r="2.3"></circle><circle cx="12.3" cy="10" r="2.3"></circle><circle cx="16.8" cy="10" r="2.3"></circle><circle cx="4.6" cy="16.2" r="2.3"></circle><circle cx="10" cy="16.2" r="2.3"></circle><circle cx="15.4" cy="16.2" r="2.3"></circle></svg></div><div><span class="wb-strip"><i class="s1 k1"></i><i class="s1 k2"></i><i class="s1 k3"></i><i class="s1 k4"></i><i class="s2 k5"></i><i class="s1 k6"></i><i class="s3 k7"></i><i class="s4 k8"></i></span></div></div>
              </div>
            </div>
            <div class="wb-pane wb-pane-rf">
              <div class="wb-card wb-info">
                <div class="ti">Refiner prompt <span class="chev">▸</span></div>
              </div>
              <div class="wb-card wb-info">
                <div class="ti">ASI fields <span class="n">6</span></div>
                <div class="wb-chips"><span>scores.sum_radii</span><span>n_circles</span><span>circles</span><span>stdout</span><span>error</span><span>validation_details</span></div>
              </div>
              <div class="wb-rf-label">Reflections</div>
              <div class="wb-rblock">
                <div class="wb-rblock-head"><span class="chev">▸</span><span class="lnk">Candidate 2</span><span class="arr">→</span><span class="lnk">Candidate 4</span><span class="bg bg-acc">accepted</span><span class="dlt">+0.06</span><span class="tok">1.9k tokens</span></div>
              </div>
              <div class="wb-rblock">
                <div class="wb-rblock-head"><span class="chev">▾</span><span class="lnk">Candidate 5</span><span class="arr">→</span><span class="lnk">Candidate 7</span><span class="bg bg-acc">accepted</span><span class="dlt">+0.03</span><span class="tok">2.4k tokens</span></div>
                <div class="wb-rblock-body">
                  <b>Reflection output</b>
                  The grid layout wastes corner space once n grows past 6. Keep the greedy placement, then anneal positions and grow radii until first contact. Cap annealing steps by n so the largest tasks stay inside the timeout.
                  <div class="quote">proposal: replace shrink_radii with anneal + grow_radii …</div>
                </div>
              </div>
              <div class="wb-rblock">
                <div class="wb-rblock-head"><span class="chev">▸</span><span class="lnk">Candidate 2</span><span class="arr">→</span><span class="lnk rej">Candidate r1</span><span class="bg bg-rej">rejected</span><span class="dlt">−0.02</span><span class="tok">2.1k tokens</span></div>
              </div>
            </div>
          </div>
          <div class="wb-dock">
            <div class="wb-dock-head">
              <span class="on">Logs</span>
              <span class="off"><img src="../static/img/gepa_logo.svg" alt="">GEPA Agent<span class="cnt">1</span></span>
            </div>
            <div class="wb-dock-body">
              <div class="wb-log"><span class="t">14:02:03</span><span class="it">iter 11</span><span class="lv-pareto">pareto</span><span>Candidate 6 joined the Pareto front (0.84)</span></div>
              <div class="wb-log"><span class="t">14:02:11</span><span class="it">iter 12</span><span class="lv-accept">accept</span><span>Candidate 7 accepted — agg score 0.87 (+0.03)</span></div>
              <div class="wb-log wb-log-new"><span class="t">14:02:14</span><span class="it">iter 13</span><span class="lv-info">info</span><span>minibatch eval started (5 tasks)&nbsp;<span class="wb-cursor"></span></span></div>
            </div>
          </div>
        </div>
      </div>
      <div class="wb-ptr wb-ptr-obs"><svg viewBox="0 0 24 24"><path d="M5 2 L5 18.8 L9.4 14.9 L12.2 21.3 L15.1 20 L12.4 13.7 L18.2 13.2 Z"></path></svg></div>
    </div>
  </div>
</div>

<!-- ================= Diagnostics & chat ================= -->

<div class="wb-section">
  <h3>Chat and Diagnostics</h3>
  <p>The GEPA Agent is built into the workbench. You can ask it questions about your runs and make edits to your configuration. It also diagnoses the run after each proposal, and when something looks wrong, it presents relevant evidence, while proposing and applying possible fixes.</p>
  <div class="wb wb-scroll">
    <div class="wb-window wb-asst">
      <div class="wb-chrome"><i></i><i></i><i></i><span>GEPA Agent</span></div>
      <div class="wb-agent-head">
        <img class="aglogo" src="../static/img/gepa_logo.svg" alt="GEPA">
        <span class="wb-agent-tab t-chat">Chat</span>
        <span class="wb-agent-tab t-diag">Diagnostics<span class="cnt">1</span></span>
        <span class="sess">＋ new chat</span>
      </div>
      <div class="wb-agent-panes">
        <div class="wb-agent-pane chat">
          <div class="wb-chat-scroll">
            <div class="wb-msg-user">Why was candidate r1 rejected?</div>
            <div class="wb-msg-think"><span class="wb-dots"><i></i><i></i><i></i></span> thinking …</div>
            <div class="wb-msg-asst">
              <img class="av" src="../static/img/gepa_logo.svg" alt="">
              <span><code>Candidate r1</code> scored <code>0.69</code> on its reflection minibatch, while its parent <code>Candidate 2</code> scored <code>0.71</code> on the same tasks. Acceptance requires beating the parent, so the proposal was not added to the pool. Its main change, a denser initial grid, slowed the search on N ≥ 7.<span class="cur"></span></span>
            </div>
          </div>
          <div class="wb-composer">
            <span class="wb-chip">@ run:circle-packing-multi ✕</span>
            <div class="wb-prompt"><span class="mark">❯ gepa</span><span class="ph">Ask a question. Use '@' to add context</span></div>
          </div>
        </div>
        <div class="wb-agent-pane diag">
          <div class="wb-diag-busy"><span class="wb-dots"><i></i><i></i><i></i></span> Diagnosing the run…</div>
          <div class="wb-diag-card">
            <div class="ti"><span class="warn">⚠</span> Timeouts on the largest tasks</div>
            <div class="fx">N=9 and N=10 hit the 60s subprocess limit in 3 of the last 5 candidates, so their scores fall to 0 and drag the aggregate down.</div>
            <div class="wb-diag-fix"><b>Suggested fix</b>Cap the local search iterations by n_circles so every task finishes inside the budget.</div>
            <div class="wb-diag-acts"><span class="acc">Apply as feedback</span><span>Dismiss</span></div>
          </div>
          <div class="wb-activity">
            <b>Agent activity</b>
            <div class="a1"><span class="ck">✓</span>Compared per task scores across candidates</div>
            <div class="a2"><span class="ck amb">✓</span>Checked stderr and timeouts in recent evals</div>
            <div class="a3"><span class="wb-dots"><i></i><i></i><i></i></span> Checking the run now</div>
          </div>
        </div>
      </div>
      <div class="wb-ptr wb-ptr-asst"><svg viewBox="0 0 24 24"><path d="M5 2 L5 18.8 L9.4 14.9 L12.2 21.3 L15.1 20 L12.4 13.7 L18.2 13.2 Z"></path></svg></div>
    </div>
  </div>
</div>

<!-- ================= Improve with feedback ================= -->

<div class="wb-section">
  <h3>Improve a candidate with feedback</h3>
  <p>Directly edit any candidate or describe what you want changed. You can evaluate the improved candidates and add them to the pool for subsequent proposals to build on.</p>
  <div class="wb wb-scroll">
    <div class="wb-improve">
      <div class="wb-imp-head"><span class="wand">✦</span> Improve Candidate 7</div>
      <div class="wb-imp-sub">Edit the candidate directly, or describe improvements as feedback for the reflection model to propose an improved candidate.</div>
      <div class="wb-imp-body">
        <div>
          <div class="wb-imp-label">program</div>
          <div class="wb-imp-code">def main(n, timeout, best):
    pts = grid_layout(n)
    r = shrink_radii(pts)
    return circles(pts, r)</div>
          <div class="wb-imp-label">Feedback for improvement</div>
          <div class="wb-imp-feedback">
            <span class="wb-type wb-type-1">Add a local search pass after placement.</span>
            <span class="wb-type wb-type-2">Cap iterations by n to avoid timeouts.</span>
          </div>
          <div class="wb-imp-btns">
            <span class="grad"><span class="wand">✦</span><b>Improve with feedback</b></span>
            <span class="ghost">Evaluate candidate</span>
          </div>
        </div>
        <div class="wb-imp-div"></div>
        <div>
          <div class="wb-imp-label">Diff between candidates</div>
          <div class="wb-imp-empty wb-de">No changes yet.</div>
          <div class="wb-diff">
            <div>  def main(n, timeout, best):</div>
            <div>      pts = grid_layout(n)</div>
            <div class="del">-     r = shrink_radii(pts)</div>
            <div class="add">+     pts, r = anneal(pts, steps_for(n))</div>
            <div class="add">+     r = grow_radii(pts, r)</div>
            <div>      return circles(pts, r)</div>
          </div>
          <div class="wb-imp-label">Evaluation</div>
          <div class="wb-eval-strip">
            <div><span>Base</span><b>0.87</b></div>
            <div><span>Improved</span><b>0.91</b></div>
            <div><span>Δ</span><b class="up">+0.04</b></div>
          </div>
        </div>
      </div>
      <div class="wb-imp-foot">
        <span class="wb-imp-btns"><span class="grad"><b>Add candidate</b></span></span>
      </div>
      <div class="wb-toast"><b>Candidate added</b>Candidate 8 joined the pool — future proposals build from it</div>
      <div class="wb-ptr wb-ptr-imp"><svg viewBox="0 0 24 24"><path d="M5 2 L5 18.8 L9.4 14.9 L12.2 21.3 L15.1 20 L12.4 13.7 L18.2 13.2 Z"></path></svg></div>
    </div>
  </div>
</div>

<!-- ================= Study recruitment ================= -->

<div class="wb-study">
  <h3>Help us shape GEPA Workbench</h3>
  <p>We are running a user study for GEPA Workbench. If you would like early access and are willing to share feedback, please sign up below!</p>
  <a class="wb-btn" href="https://forms.gle/GCD4PmjdLrKUB8Eg7" target="_blank" rel="noopener">Join the study</a>
</div>

</div>